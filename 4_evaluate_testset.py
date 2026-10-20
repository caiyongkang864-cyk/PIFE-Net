"""Reference-based evaluation on the held-out hospital Dixon test set.

This script imports the model and renderer classes directly from the training
script, preventing architecture/formula drift. Primary metrics are calculated
without per-case intensity fitting. Gain-fitted metrics are reported only as
secondary structural diagnostics because rescaling every prediction can hide a
clinically important amplitude error.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import torch
from scipy.stats import pearsonr
from tqdm import tqdm


MODALITY_CANDIDATES = {
    "W": ("01_W.nii.gz",),
    "F": ("02_F.nii.gz",),
    "IP": ("03_IP_e2_phMag.nii.gz", "03_IP_phMag.nii.gz", "03_IP.nii.gz"),
    "OP": ("04_OP_e1_phMag.nii.gz", "04_OP_phMag.nii.gz", "04_OP.nii.gz"),
}


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


def find_file(case_dir: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = case_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"{case_dir}: none of {candidates} exists")


def load_volume(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path}: expected 3D data, got {data.shape}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(data, 0.0, None).astype(np.float32, copy=False), image


def load_aligned_quartet(case_dir: Path):
    arrays = {}
    images = {}
    for name, candidates in MODALITY_CANDIDATES.items():
        arrays[name], images[name] = load_volume(find_file(case_dir, candidates))
    for name in ("F", "IP", "OP"):
        if arrays[name].shape != arrays["W"].shape:
            raise ValueError(f"{name} shape differs from W")
        if not np.allclose(images[name].affine, images["W"].affine, atol=1e-4, rtol=0):
            raise ValueError(f"{name} affine differs from W")
    return arrays, images["W"]


def w_body_mask(w: np.ndarray) -> np.ndarray:
    positive = w[w > 0]
    if positive.size == 0:
        raise ValueError("W contains no positive voxels")
    threshold = max(float(np.percentile(positive, 99.5)) * 0.01, 1e-8)
    mask = w > threshold
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
    starts = np.maximum(coords.min(0) - margin, 0)
    stops = np.minimum(coords.max(0) + margin + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(starts, stops))


def pad_tensor(x: torch.Tensor, multiple: int = 4):
    spatial = x.shape[2:]
    pads = [(multiple - size % multiple) % multiple for size in spatial]
    torch_padding = (0, pads[2], 0, pads[1], 0, pads[0])
    return torch.nn.functional.pad(x, torch_padding), spatial


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


def infer_case(
    w: np.ndarray,
    model,
    renderer,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    mask = w_body_mask(w)
    crop = bbox_from_mask(mask)
    w_crop = w[crop]
    scale = max(float(np.percentile(w[mask], 99.5)), 1e-6)
    w_tensor = torch.from_numpy((w_crop / scale)[None, None]).float().to(device)
    w_padded, original_crop_shape = pad_tensor(w_tensor, 4)

    amp_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.autocast("cuda", dtype=torch.float16)
        if use_amp
        else nullcontext()
    )
    with torch.inference_mode(), amp_context:
        latents = model(w_padded)
        rendered = renderer(
            w_padded,
            latents["fat"],
            latents["delta_frequency_hz"],
            latents["gain"],
        )

    d, h, width = original_crop_shape
    normalized = {
        "F": latents["fat"][0, 0, :d, :h, :width].float().cpu().numpy(),
        "IP": rendered["ip"][0, 0, :d, :h, :width].float().cpu().numpy(),
        "OP": rendered["op"][0, 0, :d, :h, :width].float().cpu().numpy(),
        "FrequencyHz": latents["delta_frequency_hz"][0, 0, :d, :h, :width]
        .float().cpu().numpy(),
        "Gain": latents["gain"][0, 0, :d, :h, :width].float().cpu().numpy(),
    }
    outputs = {}
    for name, cropped in normalized.items():
        full = np.zeros_like(w, dtype=np.float32)
        if name in ("F", "IP", "OP"):
            cropped = cropped * scale
        full[crop] = cropped
        if name in ("F", "IP", "OP"):
            full *= mask
        outputs[name] = full
    normalized_w = w[mask] / scale
    return outputs, {
        "scale": scale,
        "mask_voxels": int(mask.sum()),
        "W_P01_Normalized": float(np.percentile(normalized_w, 1)),
        "W_P50_Normalized": float(np.percentile(normalized_w, 50)),
        "W_P99_Normalized": float(np.percentile(normalized_w, 99)),
    }


def save_like(data: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(data.astype(np.float32), reference.affine, header)
    image.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    image.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(image, str(path))


def safe_pcc(target: np.ndarray, prediction: np.ndarray) -> float:
    if target.size < 2 or np.std(target) < 1e-8 or np.std(prediction) < 1e-8:
        return float("nan")
    return float(pearsonr(target, prediction).statistic)


def modality_metrics(
    target: np.ndarray, prediction: np.ndarray, mask: np.ndarray, prefix: str
) -> dict[str, float]:
    target = target[mask].astype(np.float64)
    prediction = prediction[mask].astype(np.float64)
    error = prediction - target
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    target_mean = float(np.mean(np.abs(target)))
    target_energy = float(np.sqrt(np.mean(target**2)))
    nmae = mae / max(target_mean, 1e-8)
    nrmse = rmse / max(target_energy, 1e-8)
    bias = float(np.mean(error)) / max(target_mean, 1e-8)
    pcc = safe_pcc(target, prediction)

    # Secondary, explicitly gain-fitted metric. Do not use it as primary score.
    gain = float(np.dot(target, prediction) / max(np.dot(prediction, prediction), 1e-8))
    fitted = gain * prediction
    fitted_nmae = float(np.mean(np.abs(fitted - target))) / max(target_mean, 1e-8)
    return {
        f"{prefix}_MAE": mae,
        f"{prefix}_NMAE": nmae,
        f"{prefix}_NRMSE": nrmse,
        f"{prefix}_RelativeBias": bias,
        f"{prefix}_PCC": pcc,
        f"{prefix}_FittedGain": gain,
        f"{prefix}_GainFittedNMAE": fitted_nmae,
    }


def bootstrap_mean_ci(values: np.ndarray, seed: int = 42) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, (5000, values.size), replace=True), axis=1)
    return tuple(float(x) for x in np.percentile(means, (2.5, 97.5)))


def evaluate(
    test_dir: str,
    checkpoint_path: str,
    training_script: str,
    output_dir: str,
    device_name: str,
) -> None:
    device = torch.device(device_name)
    training = import_training_module(training_script)
    model, renderer, protocol = build_model(training, checkpoint_path, device)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    patient_dirs = sorted(path for path in Path(test_dir).iterdir() if path.is_dir())
    for case_dir in tqdm(patient_dirs, desc="Held-out Dixon evaluation"):
        try:
            arrays, reference = load_aligned_quartet(case_dir)
            outputs, inference_info = infer_case(
                arrays["W"], model, renderer, device, device.type == "cuda"
            )
            eval_mask = (arrays["W"] + arrays["F"]) > max(
                float(np.percentile((arrays["W"] + arrays["F"])[arrays["W"] + arrays["F"] > 0], 99.5)) * 0.01,
                1e-8,
            )
            row = {"Case_ID": case_dir.name, **inference_info}
            for name in ("F", "IP", "OP"):
                row.update(modality_metrics(arrays[name], outputs[name], eval_mask, name))

            case_output = output_root / case_dir.name
            case_output.mkdir(exist_ok=True)
            for name, data in outputs.items():
                save_like(data, reference, case_output / f"{name}.nii.gz")
            rows.append(row)
        except Exception as error:
            print(f"[ERROR] {case_dir.name}: {error}")

    if not rows:
        raise RuntimeError("No test cases were evaluated successfully")
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_root / "per_case_metrics.csv", index=False)

    summary_rows = []
    for column in dataframe.select_dtypes(include=[np.number]).columns:
        values = dataframe[column].to_numpy(float)
        low, high = bootstrap_mean_ci(values)
        summary_rows.append({
            "Metric": column,
            "Mean": float(np.nanmean(values)),
            "Std": float(np.nanstd(values, ddof=1)),
            "Median": float(np.nanmedian(values)),
            "CI95_Low": low,
            "CI95_High": high,
            "N": int(np.isfinite(values).sum()),
        })
    pd.DataFrame(summary_rows).to_csv(output_root / "summary_metrics.csv", index=False)

    checkpoint_hash = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    metadata = {
        "checkpoint_sha256": checkpoint_hash,
        "test_cases_discovered": len(patient_dirs),
        "test_cases_evaluated": len(rows),
        "protocol": protocol,
        "primary_metrics_are_not_gain_fitted": True,
    }
    (output_root / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Evaluated {len(rows)}/{len(patient_dirs)} cases; results={output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-script", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        args.test_dir,
        args.checkpoint,
        args.training_script,
        args.output_dir,
        args.device,
    )
