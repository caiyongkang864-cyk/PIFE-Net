"""Generate W/F/IP/OP nnUNet input for the external MRISegmenter dataset.

Only T1w-FS/W is used to determine normalization, body crop and inference
geometry. The downstream ``_0000`` W file is copied byte-for-byte from the
source dataset so it remains identical to the W-only baseline. Labels are
copied after strict grid validation and never influence generation. Model and
renderer definitions are imported from the exact training script to prevent
formula or architecture drift.

The resulting F/IP/OP are physics-guided synthetic priors in the hospital
training protocol domain. They are not claimed to be measured Dixon sequences
from the MRISegmenter scanner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
from tqdm import tqdm


def import_training_module(path: str):
    module_path = Path(path).resolve()
    name = "dixon_training_definition"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import training definitions from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_model(training, checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint.get("config", {})
    model = training.PhysicsParameterNet(
        base_c=int(config.get("base_channels", 32)),
        use_coordinates=bool(config.get("use_coordinates", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    renderer = training.DixonMagnitudeRenderer(
        te_op_ms=float(config.get("te_op_ms", 1.151)),
        te_ip_ms=float(config.get("te_ip_ms", 2.302)),
        t2_star_w_ms=float(config.get("t2_star_w_ms", 25.0)),
        t2_star_f_ms=float(config.get("t2_star_f_ms", 20.0)),
    ).to(device)
    renderer.eval()
    return model, renderer, config


def load_w(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path}: expected 3D data, got {data.shape}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(data, 0.0, None).astype(np.float32, copy=False), image


def body_mask_from_w(w: np.ndarray) -> np.ndarray:
    positive = w[w > 0]
    if positive.size == 0:
        raise ValueError("input contains no positive W signal")
    robust_peak = float(np.percentile(positive, 99.5))
    mask = w > max(robust_peak * 0.01, 1e-8)
    mask = ndi.binary_closing(mask, iterations=2)
    mask = ndi.binary_fill_holes(mask)
    labels, count = ndi.label(mask)
    if count:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        minimum = max(32, int(sizes.max() * 0.005))
        mask = np.isin(labels, np.flatnonzero(sizes >= minimum))
    return ndi.binary_dilation(mask, iterations=2)


def bbox_from_mask(mask: np.ndarray, margin: int = 8) -> tuple[slice, slice, slice]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("body mask is empty")
    starts = np.maximum(coords.min(axis=0) - margin, 0)
    stops = np.minimum(coords.max(axis=0) + margin + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(starts, stops))


def pad_tensor(x: torch.Tensor, multiple: int = 4):
    original_shape = x.shape[2:]
    pads = [(multiple - size % multiple) % multiple for size in original_shape]
    return (
        torch.nn.functional.pad(x, (0, pads[2], 0, pads[1], 0, pads[0])),
        original_shape,
    )


def infer(
    w: np.ndarray,
    model,
    renderer,
    device: torch.device,
    margin: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    mask = body_mask_from_w(w)
    crop = bbox_from_mask(mask, margin)
    scale = max(float(np.percentile(w[mask], 99.5)), 1e-6)
    crop_array = w[crop]
    tensor = torch.from_numpy((crop_array / scale)[None, None]).float().to(device)
    padded, original_crop_shape = pad_tensor(tensor, 4)

    use_amp = device.type == "cuda"
    amp_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.autocast("cuda", dtype=torch.float16)
        if use_amp
        else nullcontext()
    )
    with torch.inference_mode(), amp_context:
        latents = model(padded)
        rendered = renderer(
            padded,
            latents["fat"],
            latents["delta_frequency_hz"],
            latents["gain"],
        )

    d, h, width = original_crop_shape
    cropped_outputs = {
        "F": latents["fat"][0, 0, :d, :h, :width].float().cpu().numpy() * scale,
        "IP": rendered["ip"][0, 0, :d, :h, :width].float().cpu().numpy() * scale,
        "OP": rendered["op"][0, 0, :d, :h, :width].float().cpu().numpy() * scale,
        "FrequencyHz": latents["delta_frequency_hz"][0, 0, :d, :h, :width]
        .float().cpu().numpy(),
        "Gain": latents["gain"][0, 0, :d, :h, :width].float().cpu().numpy(),
    }
    # W is used here only as the inference source. The downstream W channel is
    # copied directly from the original NIfTI file in prepare_downstream(), so
    # masking/normalization can never modify the baseline input.
    outputs: dict[str, np.ndarray] = {}
    for name, value in cropped_outputs.items():
        full = np.zeros_like(w, dtype=np.float32)
        full[crop] = value
        if name in ("F", "IP", "OP"):
            full *= mask
        outputs[name] = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)

    info = {
        "Scale": scale,
        "BodyVoxels": int(mask.sum()),
        "CropShape": "x".join(str(x) for x in original_crop_shape),
        "Orientation": "".join(nib.aff2axcodes(np.eye(4))),
        "W_P01_Normalized": float(np.percentile(w[mask] / scale, 1)),
        "W_P50_Normalized": float(np.percentile(w[mask] / scale, 50)),
        "W_P99_Normalized": float(np.percentile(w[mask] / scale, 99)),
        "FrequencySaturationRate": float(
            np.mean(np.abs(cropped_outputs["FrequencyHz"]) >= 79.9)
        ),
        "GainP01": float(np.percentile(cropped_outputs["Gain"], 1)),
        "GainP99": float(np.percentile(cropped_outputs["Gain"], 99)),
    }
    return outputs, info


def save_like(data: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(data.astype(np.float32), reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    temporary = path.with_name(path.name + ".tmp.nii.gz")
    nib.save(image, str(temporary))
    os.replace(temporary, path)


def validate_label_grid(label_path: Path, reference: nib.Nifti1Image) -> None:
    label = nib.load(str(label_path))
    if label.shape[:3] != reference.shape[:3]:
        raise ValueError(f"label shape {label.shape} != image shape {reference.shape}")
    if not np.allclose(label.affine, reference.affine, atol=1e-4, rtol=0):
        raise ValueError("label affine differs from input W affine")


def update_dataset_json(source_root: Path, target_root: Path, num_training: int) -> None:
    source_json = source_root / "dataset.json"
    metadata = json.loads(source_json.read_text()) if source_json.is_file() else {}
    metadata["channel_names"] = {"0": "W", "1": "synthetic_F", "2": "synthetic_IP", "3": "synthetic_OP"}
    metadata["numTraining"] = num_training
    metadata.setdefault("file_ending", ".nii.gz")
    if "labels" not in metadata:
        raise ValueError(
            "Source dataset.json with the authoritative MRISegmenter label mapping is required"
        )
    (target_root / "dataset.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def prepare_downstream(
    source_dir: str,
    target_dir: str,
    checkpoint_path: str,
    training_script: str,
    device_name: str,
    margin: int = 8,
) -> None:
    source_root = Path(source_dir)
    target_root = Path(target_dir)
    if source_root.resolve() == target_root.resolve():
        raise ValueError("source-dir and target-dir must be different directories")
    source_images = source_root / "imagesTs"
    source_labels = source_root / "labelsTs"
    output_images = target_root / "imagesTs"
    output_labels = target_root / "labelsTs"
    diagnostics = target_root / "physics_diagnostics"
    for directory in (output_images, output_labels, diagnostics):
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_name)
    training = import_training_module(training_script)
    model, renderer, protocol = build_model(training, checkpoint_path, device)
    inputs = sorted(source_images.glob("*_0000.nii.gz"))
    if not inputs:
        raise RuntimeError(f"No *_0000.nii.gz files found in {source_images}")

    rows = []
    failures = []
    for input_path in tqdm(inputs, desc="MRISegmenter physics-prior generation"):
        case_name = input_path.name.removesuffix("_0000.nii.gz")
        label_path = source_labels / f"{case_name}.nii.gz"
        try:
            if not label_path.is_file():
                raise FileNotFoundError(f"missing exact label {label_path.name}")
            w, reference = load_w(input_path)
            validate_label_grid(label_path, reference)
            outputs, info = infer(w, model, renderer, device, margin)

            # Preserve the baseline W/T1w-FS exactly: no dtype conversion,
            # clipping, masking, header reconstruction or NIfTI recompression.
            shutil.copy2(
                input_path,
                output_images / f"{case_name}_0000.nii.gz",
            )

            channels = {"F": "0001", "IP": "0002", "OP": "0003"}
            for name, channel in channels.items():
                save_like(
                    outputs[name], reference, output_images / f"{case_name}_{channel}.nii.gz"
                )
            save_like(
                outputs["FrequencyHz"], reference,
                diagnostics / f"{case_name}_frequency_hz.nii.gz",
            )
            save_like(
                outputs["Gain"], reference, diagnostics / f"{case_name}_gain.nii.gz"
            )
            shutil.copy2(label_path, output_labels / label_path.name)
            info.update({
                "Case_ID": case_name,
                "Orientation": "".join(nib.aff2axcodes(reference.affine)),
                "InputShape": "x".join(str(x) for x in w.shape),
            })
            rows.append(info)
        except Exception as error:
            failures.append({"Case_ID": case_name, "Error": str(error)})
            print(f"[ERROR] {case_name}: {error}")

    if not rows:
        raise RuntimeError("No downstream cases were generated successfully")
    update_dataset_json(source_root, target_root, len(rows))
    pd.DataFrame(rows).to_csv(target_root / "generation_manifest.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(target_root / "generation_failures.csv", index=False)

    metadata = {
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "source_dataset": str(source_root.resolve()),
        "cases_discovered": len(inputs),
        "cases_generated": len(rows),
        "cases_failed": len(failures),
        "training_protocol": protocol,
        "interpretation": "physics-guided synthetic priors, not measured Dixon",
        "normalization": (
            "Original W copied byte-for-byte; W-only foreground p99.5 used "
            "internally for synthesis; one shared scale restored F/IP/OP"
        ),
    }
    (target_root / "generation_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Generated {len(rows)}/{len(inputs)} cases in {target_root}")
    if failures:
        raise RuntimeError(f"{len(failures)} cases failed; see generation_failures.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-script", required=True)
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_downstream(
        args.source_dir,
        args.target_dir,
        args.checkpoint,
        args.training_script,
        args.device,
        args.margin,
    )
