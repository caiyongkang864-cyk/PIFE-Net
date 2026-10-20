"""Convert aligned augmented Dixon quartets to compact training tensors.

Data contract written by this script:
    W, F, IP, OP: [1, D, H, W], float16, one shared intensity scale
    mask:         [1, D, H, W], bool, valid foreground only
    scale:        scalar float32, original common intensity scale
    crop_bbox:    six int64 values in the original voxel grid
    original_shape: three int64 values

No pseudo T2* or phase target is estimated here. Those quantities are not
identifiable from magnitude images under the earlier simplified inversion and
were contaminating supervision. Physics is enforced by the differentiable
renderer during training.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm


SUFFIXES = {
    "W": "01_W.nii.gz",
    "F": "02_F.nii.gz",
    "IP": "03_IP.nii.gz",
    "OP": "04_OP.nii.gz",
}


def load_3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path}: expected 3D data, got {data.shape}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(data, 0.0, None).astype(np.float32, copy=False), image.affine


def load_and_validate_quartet(case_dir: Path, prefix: str) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    affines: dict[str, np.ndarray] = {}
    for modality, suffix in SUFFIXES.items():
        arrays[modality], affines[modality] = load_3d(case_dir / f"{prefix}_{suffix}")

    shape = arrays["W"].shape
    affine = affines["W"]
    for modality in ("F", "IP", "OP"):
        if arrays[modality].shape != shape:
            raise ValueError(
                f"{case_dir}/{prefix}: {modality} shape {arrays[modality].shape} "
                f"!= W shape {shape}"
            )
        if not np.allclose(affines[modality], affine, rtol=1e-5, atol=1e-4):
            raise ValueError(f"{case_dir}/{prefix}: {modality} affine != W affine")
    return arrays


def build_foreground_mask(w: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Use the W+F component image so fat-dominant tissue is not cropped out."""
    component_sum = w + f
    positive = component_sum[component_sum > 0]
    if positive.size == 0:
        raise ValueError("W+F contains no positive foreground")
    robust_peak = float(np.percentile(positive, 99.5))
    threshold = max(robust_peak * 0.01, np.finfo(np.float32).eps)
    mask = component_sum > threshold
    if not np.any(mask):
        raise ValueError("foreground mask is empty")
    return mask


def bounding_box(mask: np.ndarray, margin: int) -> tuple[int, int, int, int, int, int]:
    coords = np.argwhere(mask)
    starts = np.maximum(coords.min(axis=0) - margin, 0)
    stops = np.minimum(coords.max(axis=0) + margin + 1, mask.shape)
    return (
        int(starts[0]), int(stops[0]),
        int(starts[1]), int(stops[1]),
        int(starts[2]), int(stops[2]),
    )


def crop_array(array: np.ndarray, bbox: tuple[int, ...]) -> np.ndarray:
    x1, x2, y1, y2, z1, z2 = bbox
    return array[x1:x2, y1:y2, z1:z2]


def pad_to_multiple(array: np.ndarray, multiple: int, value: float | bool) -> np.ndarray:
    padding = tuple((0, (multiple - size % multiple) % multiple) for size in array.shape)
    return np.pad(array, padding, mode="constant", constant_values=value)


def common_intensity_scale(w: np.ndarray, f: np.ndarray, mask: np.ndarray) -> float:
    """One shared scale preserves cross-modality amplitude relationships."""
    values = (w + f)[mask]
    return max(float(np.percentile(values, 99.5)), 1e-6)


def process_variant(
    case_dir: Path,
    prefix: str,
    output_dir: Path,
    margin: int = 4,
    size_multiple: int = 4,
) -> dict[str, object]:
    arrays = load_and_validate_quartet(case_dir, prefix)
    original_shape = arrays["W"].shape
    full_mask = build_foreground_mask(arrays["W"], arrays["F"])
    bbox = bounding_box(full_mask, margin)

    cropped = {name: crop_array(array, bbox) for name, array in arrays.items()}
    mask = crop_array(full_mask, bbox)
    scale = common_intensity_scale(cropped["W"], cropped["F"], mask)

    tensors: dict[str, object] = {}
    for name, array in cropped.items():
        normalized = np.clip(array / scale, 0.0, 20.0)
        normalized = pad_to_multiple(normalized, size_multiple, 0.0)
        tensors[name] = torch.from_numpy(normalized[None]).to(torch.float16)

    mask = pad_to_multiple(mask, size_multiple, False)
    tensors["mask"] = torch.from_numpy(mask[None].copy()).bool()
    tensors["scale"] = torch.tensor(scale, dtype=torch.float32)
    tensors["crop_bbox"] = torch.tensor(bbox, dtype=torch.int64)
    tensors["original_shape"] = torch.tensor(original_shape, dtype=torch.int64)
    tensors["case_id"] = f"{case_dir.name}/{prefix}"

    output_path = output_dir / f"{case_dir.name}_{prefix}.pt"
    temporary_path = output_path.with_suffix(".pt.tmp")
    torch.save(tensors, temporary_path)
    os.replace(temporary_path, output_path)
    return {
        "ok": True,
        "case": tensors["case_id"],
        "output": str(output_path),
        "scale": scale,
    }


def discover_tasks(input_dir: Path, output_dir: Path) -> list[tuple[Path, str, Path]]:
    tasks = []
    suffix = f"_{SUFFIXES['W']}"
    for case_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        prefixes = sorted(
            path.name[:-len(suffix)]
            for path in case_dir.iterdir()
            if path.name.endswith(suffix)
        )
        tasks.extend((case_dir, prefix, output_dir) for prefix in prefixes)
    return tasks


def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    workers: int | None = None,
    margin: int = 4,
    size_multiple: int = 4,
) -> None:
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = discover_tasks(input_root, output_root)
    if not tasks:
        raise RuntimeError(f"No W/F/IP/OP variants found under {input_root}")

    max_workers = workers or min(8, os.cpu_count() or 1)
    failures: list[str] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                process_variant, case_dir, prefix, out, margin, size_multiple
            ): f"{case_dir.name}/{prefix}"
            for case_dir, prefix, out in tasks
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_map),
            total=len(future_map),
            desc="Dixon tensor preprocessing",
        ):
            case_id = future_map[future]
            try:
                future.result()
            except Exception as error:
                failures.append(f"{case_id}: {error}")

    print(f"Completed {len(tasks) - len(failures)}/{len(tasks)} variants")
    if failures:
        raise RuntimeError(
            f"{len(failures)} variants failed. First failures: " + "; ".join(failures[:5])
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Augmented patient-folder root")
    parser.add_argument("--output", required=True, help="Flat output directory for PT files")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--margin", type=int, default=4)
    parser.add_argument("--size-multiple", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_dataset(
        input_dir=args.input,
        output_dir=args.output,
        workers=args.workers,
        margin=args.margin,
        size_multiple=args.size_multiple,
    )
