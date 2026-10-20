"""Physics-preserving offline augmentation for aligned Dixon volumes.

Reads a real aligned W/F/IP/OP quartet and creates variants using one smooth
positive multiplicative field shared by all modalities. Dixon's magnitude
equation is homogeneous of degree one, so this changes W while preserving the
voxelwise water/fat relationship. Hidden TE/TR/fat parameters are deliberately
not randomized: a deterministic W-only model cannot recover unobserved random
acquisition variables when identical W inputs have different targets.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import random
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


MODALITY_CANDIDATES = {
    "W": ("01_W.nii.gz",),
    "F": ("02_F.nii.gz",),
    "IP": ("03_IP_e2_phMag.nii.gz", "03_IP_phMag.nii.gz", "03_IP.nii.gz"),
    "OP": ("04_OP_e1_phMag.nii.gz", "04_OP_phMag.nii.gz", "04_OP.nii.gz"),
}


@dataclass(frozen=True)
class Volume:
    data: np.ndarray
    affine: np.ndarray
    header: nib.Nifti1Header


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_file(case_dir: Path, candidates: tuple[str, ...]) -> Path:
    for name in candidates:
        path = case_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"{case_dir}: none of {candidates} exists")


def load_volume(path: Path) -> Volume:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path}: expected 3D data, got shape={data.shape}")
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.clip(data, 0.0, None).astype(np.float32, copy=False)
    return Volume(data, image.affine.copy(), image.header.copy())


def load_aligned_quartet(case_dir: Path) -> dict[str, Volume]:
    volumes = {
        key: load_volume(find_file(case_dir, names))
        for key, names in MODALITY_CANDIDATES.items()
    }
    reference = volumes["W"]
    for name, volume in volumes.items():
        if volume.data.shape != reference.data.shape:
            raise ValueError(
                f"{case_dir}: {name} shape {volume.data.shape} != "
                f"W shape {reference.data.shape}; resample before augmentation"
            )
        if not np.allclose(volume.affine, reference.affine, rtol=1e-5, atol=1e-4):
            raise ValueError(
                f"{case_dir}: {name} affine differs from W; resample first"
            )
    return volumes


def make_shared_smooth_field(
    shape: tuple[int, int, int],
    generator: torch.Generator,
    strength: float,
    global_scale_range: tuple[float, float],
) -> np.ndarray:
    """Create a positive low-frequency field with median near one."""
    low_shape = tuple(max(2, min(5, int(np.ceil(size / 48)))) for size in shape)
    noise = torch.randn((1, 1, *low_shape), generator=generator)
    noise = F.interpolate(noise, size=shape, mode="trilinear", align_corners=False)
    noise = noise - noise.mean()
    noise = noise / noise.std().clamp_min(1e-6)

    field = torch.exp(strength * noise)
    field = field / field.median().clamp_min(1e-6)
    low, high = global_scale_range
    global_scale = low + (high - low) * torch.rand((), generator=generator)
    return (field * global_scale).squeeze().numpy().astype(np.float32)


def save_nifti(data: np.ndarray, reference: Volume, path: Path) -> None:
    image = nib.Nifti1Image(
        np.asarray(data, dtype=np.float32),
        reference.affine,
        reference.header.copy(),
    )
    image.set_data_dtype(np.float32)
    nib.save(image, str(path))


def save_variant(
    quartet: dict[str, Volume],
    field: np.ndarray,
    output_dir: Path,
    patient_id: str,
    variant_index: int,
    io_pool: concurrent.futures.ThreadPoolExecutor,
) -> list[concurrent.futures.Future]:
    prefix = f"{patient_id}_variant_{variant_index:02d}"
    reference = quartet["W"]
    suffixes = {
        "W": "01_W.nii.gz",
        "F": "02_F.nii.gz",
        "IP": "03_IP.nii.gz",
        "OP": "04_OP.nii.gz",
    }
    futures = []
    for modality, suffix in suffixes.items():
        # The same field is essential. Different fields would break the
        # voxelwise Dixon relationship.
        augmented = quartet[modality].data * field
        path = output_dir / f"{prefix}_{suffix}"
        futures.append(io_pool.submit(save_nifti, augmented, reference, path))
    return futures


def process_case(
    case_dir: Path,
    output_root: Path,
    samples_per_patient: int,
    seed: int,
    field_strength: float,
    global_scale_range: tuple[float, float],
    io_pool: concurrent.futures.ThreadPoolExecutor,
) -> None:
    quartet = load_aligned_quartet(case_dir)
    case_output = output_root / case_dir.name
    case_output.mkdir(parents=True, exist_ok=True)

    futures: list[concurrent.futures.Future] = []
    shape = quartet["W"].data.shape
    patient_hash = sum(case_dir.name.encode("utf-8")) * 1009
    for variant_index in range(samples_per_patient):
        if variant_index == 0:
            field = np.ones(shape, dtype=np.float32)
        else:
            variant_seed = seed + patient_hash + variant_index
            generator = torch.Generator(device="cpu").manual_seed(variant_seed)
            field = make_shared_smooth_field(
                shape, generator, field_strength, global_scale_range
            )
        futures.extend(
            save_variant(
                quartet, field, case_output, case_dir.name, variant_index, io_pool
            )
        )

    # result() exposes compression/disk errors rather than silently accepting
    # an incomplete W/F/IP/OP quartet.
    for future in futures:
        future.result()


def augment_dataset(
    real_dataset_dir: str,
    output_dir: str,
    subset_size: int | None = 950,
    samples_per_patient: int = 10,
    seed: int = 42,
    field_strength: float = 0.15,
    global_scale_range: tuple[float, float] = (0.90, 1.10),
    io_workers: int = 8,
) -> None:
    if samples_per_patient < 1:
        raise ValueError("samples_per_patient must be at least 1")
    if field_strength < 0:
        raise ValueError("field_strength must be non-negative")
    if not 0 < global_scale_range[0] <= global_scale_range[1]:
        raise ValueError("global_scale_range must be positive and ordered")

    seed_everything(seed)
    input_root = Path(real_dataset_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    patients = sorted(path for path in input_root.iterdir() if path.is_dir())
    if subset_size is not None and subset_size < len(patients):
        patients = random.Random(seed).sample(patients, subset_size)
        patients.sort()

    failures: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as io_pool:
        for case_dir in tqdm(patients, desc="Physics-preserving augmentation", unit="case"):
            try:
                process_case(
                    case_dir,
                    output_root,
                    samples_per_patient,
                    seed,
                    field_strength,
                    global_scale_range,
                    io_pool,
                )
            except Exception as error:
                failures.append((case_dir.name, str(error)))
                print(f"[ERROR] {case_dir.name}: {error}")

    print(
        f"Completed {len(patients) - len(failures)}/{len(patients)} patients; "
        f"output={output_root}"
    )
    if failures:
        examples = "; ".join(f"{case}: {message}" for case, message in failures[:5])
        raise RuntimeError(f"{len(failures)} patients failed. First: {examples}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Root containing patient folders")
    parser.add_argument("--output", required=True)
    parser.add_argument("--subset-size", type=int, default=950)
    parser.add_argument("--samples-per-patient", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--field-strength", type=float, default=0.15)
    parser.add_argument("--io-workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    augment_dataset(
        real_dataset_dir=args.input,
        output_dir=args.output,
        subset_size=args.subset_size,
        samples_per_patient=args.samples_per_patient,
        seed=args.seed,
        field_strength=args.field_strength,
        io_workers=args.io_workers,
    )
