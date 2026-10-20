"""No-reference quality control for MRISegmenter synthetic Dixon priors.

These metrics test numerical validity, renderer closure, latent saturation and
expected water-fat cancellation behavior. They do NOT establish that synthetic
F/IP/OP equal unobserved real Dixon acquisitions. Semantic value must ultimately
be assessed by held-out downstream segmentation performance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_3d(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path}: expected 3D data, got {data.shape}")
    return data, image


def validate_grid(reference: nib.Nifti1Image, image: nib.Nifti1Image, name: str):
    if image.shape[:3] != reference.shape[:3]:
        raise ValueError(f"{name}: shape differs from W")
    if not np.allclose(image.affine, reference.affine, atol=1e-4, rtol=0):
        raise ValueError(f"{name}: affine differs from W")


def finite_nonnegative_metrics(array: np.ndarray, mask: np.ndarray, prefix: str):
    values = array[mask]
    return {
        f"{prefix}_FiniteRate": float(np.mean(np.isfinite(values))),
        f"{prefix}_NegativeRate": float(np.mean(values < 0)),
        f"{prefix}_P01": float(np.nanpercentile(values, 1)),
        f"{prefix}_P50": float(np.nanpercentile(values, 50)),
        f"{prefix}_P99": float(np.nanpercentile(values, 99)),
    }


def gradient_per_mm(field: np.ndarray, mask: np.ndarray, affine: np.ndarray) -> float:
    spacing = nib.affines.voxel_sizes(affine)
    values = []
    for axis in range(3):
        difference = np.abs(np.diff(field, axis=axis)) / max(float(spacing[axis]), 1e-8)
        left = np.take(mask, indices=range(mask.shape[axis] - 1), axis=axis)
        right = np.take(mask, indices=range(1, mask.shape[axis]), axis=axis)
        valid = left & right
        if np.any(valid):
            values.append(float(np.mean(difference[valid])))
    return float(np.mean(values)) if values else float("nan")


def renderer_closure(
    w: np.ndarray,
    fat: np.ndarray,
    ip: np.ndarray,
    op: np.ndarray,
    frequency_hz: np.ndarray,
    gain: np.ndarray,
    mask: np.ndarray,
    protocol: dict,
) -> dict[str, float]:
    te_op = float(protocol.get("te_op_ms", 1.151))
    te_ip = float(protocol.get("te_ip_ms", 2.302))
    t2_w = float(protocol.get("t2_star_w_ms", 25.0))
    t2_f = float(protocol.get("t2_star_f_ms", 20.0))
    nominal_hz = 1.0 / (2.0 * te_op * 1e-3)

    def render(te_ms: float):
        w_echo = w * np.exp(-te_ms / t2_w)
        f_echo = fat * np.exp(-te_ms / t2_f)
        phase = 2.0 * math.pi * (nominal_hz + frequency_hz) * te_ms * 1e-3
        power = w_echo**2 + f_echo**2 + 2.0 * w_echo * f_echo * np.cos(phase)
        return gain * np.sqrt(np.clip(power, 1e-8, None))

    expected_ip = render(te_ip)
    expected_op = render(te_op)
    scale = max(float(np.mean(np.abs(ip[mask]) + np.abs(op[mask]))) * 0.5, 1e-8)
    return {
        "IP_RendererClosureNMAE": float(np.mean(np.abs(ip[mask] - expected_ip[mask]))) / scale,
        "OP_RendererClosureNMAE": float(np.mean(np.abs(op[mask] - expected_op[mask]))) / scale,
    }


def cancellation_metrics(
    w: np.ndarray,
    fat: np.ndarray,
    ip: np.ndarray,
    op: np.ndarray,
    body_mask: np.ndarray,
) -> dict[str, float]:
    fraction = fat / np.clip(w + fat, 1e-8, None)
    mixed = body_mask & (fraction >= 0.10) & (fraction <= 0.90)
    if not np.any(mixed):
        return {
            "MixedVoxelRate": 0.0,
            "Mixed_IPminusOP_Normalized": float("nan"),
            "Mixed_OPgreaterIP_Rate": float("nan"),
        }
    contrast = (ip[mixed] - op[mixed]) / np.clip(ip[mixed] + op[mixed], 1e-8, None)
    return {
        "MixedVoxelRate": float(np.mean(mixed[body_mask])),
        "Mixed_IPminusOP_Normalized": float(np.mean(contrast)),
        "Mixed_OPgreaterIP_Rate": float(np.mean(op[mixed] > ip[mixed])),
    }


def label_fat_statistics(
    case_name: str,
    w: np.ndarray,
    fat: np.ndarray,
    labels: np.ndarray,
    label_names: dict[int, str],
) -> list[dict[str, object]]:
    fraction = fat / np.clip(w + fat, 1e-8, None)
    rows = []
    for label_id in np.unique(labels):
        label_id = int(label_id)
        if label_id == 0:
            continue
        mask = labels == label_id
        if not np.any(mask):
            continue
        rows.append({
            "Case_ID": case_name,
            "Label_ID": label_id,
            "Label_Name": label_names.get(label_id, f"label_{label_id}"),
            "Voxels": int(mask.sum()),
            "FatFraction_Mean": float(np.mean(fraction[mask])),
            "FatFraction_Median": float(np.median(fraction[mask])),
            "FatFraction_P90": float(np.percentile(fraction[mask], 90)),
        })
    return rows


def bootstrap_ci(values: np.ndarray, seed: int = 42):
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, (5000, values.size), replace=True), axis=1)
    return tuple(float(x) for x in np.percentile(means, (2.5, 97.5)))


def evaluate(dataset_dir: str, output_dir: str) -> None:
    root = Path(dataset_dir)
    images = root / "imagesTr"
    labels_dir = root / "labelsTr"
    diagnostics = root / "physics_diagnostics"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset_json = json.loads((root / "dataset.json").read_text())
    label_names = {int(value): name for name, value in dataset_json["labels"].items()}
    generation_metadata = json.loads((root / "generation_metadata.json").read_text())
    protocol = generation_metadata.get("training_protocol", {})

    case_rows = []
    label_rows = []
    failures = []
    w_paths = sorted(images.glob("*_0000.nii.gz"))
    for w_path in tqdm(w_paths, desc="Synthetic-prior quality control"):
        case_name = w_path.name.removesuffix("_0000.nii.gz")
        paths = {
            "F": images / f"{case_name}_0001.nii.gz",
            "IP": images / f"{case_name}_0002.nii.gz",
            "OP": images / f"{case_name}_0003.nii.gz",
            "FrequencyHz": diagnostics / f"{case_name}_frequency_hz.nii.gz",
            "Gain": diagnostics / f"{case_name}_gain.nii.gz",
            "Label": labels_dir / f"{case_name}.nii.gz",
        }
        try:
            if not all(path.is_file() for path in paths.values()):
                missing = [name for name, path in paths.items() if not path.is_file()]
                raise FileNotFoundError(f"missing {missing}")
            w, reference = load_3d(w_path)
            arrays = {}
            for name, path in paths.items():
                arrays[name], image = load_3d(path)
                validate_grid(reference, image, name)
            labels = np.rint(arrays.pop("Label")).astype(np.int32)

            finite_w = np.isfinite(w) & (w >= 0)
            positive = w[finite_w & (w > 0)]
            threshold = max(float(np.percentile(positive, 99.5)) * 0.01, 1e-8)
            body = finite_w & ((w + arrays["F"]) > threshold)
            if not np.any(body):
                raise ValueError("empty body mask")

            row = {"Case_ID": case_name, "BodyVoxels": int(body.sum())}
            for name in ("F", "IP", "OP"):
                row.update(finite_nonnegative_metrics(arrays[name], body, name))
            row.update(
                renderer_closure(
                    w, arrays["F"], arrays["IP"], arrays["OP"],
                    arrays["FrequencyHz"], arrays["Gain"], body, protocol,
                )
            )
            row.update(cancellation_metrics(w, arrays["F"], arrays["IP"], arrays["OP"], body))
            row.update({
                "FrequencyHz_P01": float(np.percentile(arrays["FrequencyHz"][body], 1)),
                "FrequencyHz_P99": float(np.percentile(arrays["FrequencyHz"][body], 99)),
                "FrequencySaturationRate": float(np.mean(np.abs(arrays["FrequencyHz"][body]) >= 79.9)),
                "FrequencyGradient_HzPerMM": gradient_per_mm(arrays["FrequencyHz"], body, reference.affine),
                "Gain_P01": float(np.percentile(arrays["Gain"][body], 1)),
                "Gain_P99": float(np.percentile(arrays["Gain"][body], 99)),
                "GainGradient_PerMM": gradient_per_mm(arrays["Gain"], body, reference.affine),
            })
            case_rows.append(row)
            label_rows.extend(
                label_fat_statistics(case_name, w, arrays["F"], labels, label_names)
            )
        except Exception as error:
            failures.append({"Case_ID": case_name, "Error": str(error)})
            print(f"[ERROR] {case_name}: {error}")

    if not case_rows:
        raise RuntimeError("No cases passed quality control")
    case_df = pd.DataFrame(case_rows)
    case_df.to_csv(output / "per_case_physics_qc.csv", index=False)
    pd.DataFrame(label_rows).to_csv(output / "per_label_fat_statistics.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(output / "physics_qc_failures.csv", index=False)

    summaries = []
    for column in case_df.select_dtypes(include=[np.number]).columns:
        values = case_df[column].to_numpy(float)
        low, high = bootstrap_ci(values)
        summaries.append({
            "Metric": column,
            "Mean": float(np.nanmean(values)),
            "Std": float(np.nanstd(values, ddof=1)),
            "Median": float(np.nanmedian(values)),
            "CI95_Low": low,
            "CI95_High": high,
            "N": int(np.isfinite(values).sum()),
        })
    pd.DataFrame(summaries).to_csv(output / "physics_qc_summary.csv", index=False)

    interpretation = {
        "cases_evaluated": len(case_rows),
        "cases_failed": len(failures),
        "important_limitation": (
            "No-reference QC verifies internal consistency and plausibility only; "
            "it cannot prove agreement with unobserved real MRISegmenter Dixon images."
        ),
        "removed_invalid_claims": [
            "all labeled organs should have near-zero fat",
            "every organ boundary should show an opposed-phase black line",
            "fixed universal OFL/CSBC thresholds",
        ],
    }
    (output / "interpretation.json").write_text(
        json.dumps(interpretation, indent=2), encoding="utf-8"
    )
    print(f"QC completed for {len(case_rows)}/{len(w_paths)} cases; output={output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.dataset_dir, args.output_dir)
