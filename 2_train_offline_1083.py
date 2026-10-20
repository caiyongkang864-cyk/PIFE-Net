"""Train a W-only, physics-guided Dixon synthesis network.

The neural network predicts three latent fields from W. Normalized spatial
coordinates are optional and disabled by default for cross-dataset transfer:
    F_hat              non-negative fat amplitude
    delta_frequency_hz residual water-fat frequency field
    gain               bounded positive receive/reconstruction gain

A fixed differentiable Dixon renderer then produces IP_hat and OP_hat at the
protocol echo times. F/IP/OP supervision is computed only inside the valid mask
saved by ``3_preprocess_to_pt``. The renderer uses one unified magnitude
equation for both echoes; in-phase/opposed-phase behavior comes from TE-dependent
phase, not from manually switching a plus/minus sign.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

# Must be set before PyTorch initializes CUDA. This reduces allocator
# fragmentation when volumes (especially full-volume validation cases) differ
# in shape.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3DBackbone(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 3, base_c: int = 32):
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, base_c)
        self.enc2 = ConvBlock3D(base_c, base_c * 2, stride=2)
        self.enc3 = ConvBlock3D(base_c * 2, base_c * 4, stride=2)
        self.bottleneck = ConvBlock3D(base_c * 4, base_c * 4)
        self.dec2 = ConvBlock3D(base_c * 4 + base_c * 2, base_c * 2)
        self.dec1 = ConvBlock3D(base_c * 2 + base_c, base_c)
        self.output = nn.Conv3d(base_c, out_channels, kernel_size=1)

    def _run(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.training and torch.is_grad_enabled():
            return checkpoint(module, x, use_reentrant=False)
        return module(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self._run(self.enc1, x)
        e2 = self._run(self.enc2, e1)
        e3 = self._run(self.enc3, e2)
        bottleneck = self._run(self.bottleneck, e3)

        up2 = F.interpolate(
            bottleneck, size=e2.shape[2:], mode="trilinear", align_corners=False
        )
        d2 = self._run(self.dec2, torch.cat((up2, e2), dim=1))
        up1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self._run(self.dec1, torch.cat((up1, e1), dim=1))
        return self.output(d1)


class PhysicsParameterNet(nn.Module):
    """Predict identifiable task latents without directly regressing IP/OP."""

    def __init__(
        self,
        base_c: int = 32,
        max_residual_frequency_hz: float = 80.0,
        max_log_gain: float = 0.40,
        use_coordinates: bool = False,
    ):
        super().__init__()
        self.use_coordinates = bool(use_coordinates)
        self.unet = UNet3DBackbone(4 if self.use_coordinates else 1, 3, base_c)
        self.max_residual_frequency_hz = float(max_residual_frequency_hz)
        self.max_log_gain = float(max_log_gain)

    @staticmethod
    def coordinate_channels(x: torch.Tensor) -> torch.Tensor:
        batch, _, depth, height, width = x.shape
        z = torch.linspace(-1, 1, depth, device=x.device, dtype=x.dtype)
        y = torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype)
        x_axis = torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype)
        z = z.view(1, 1, depth, 1, 1).expand(batch, 1, depth, height, width)
        y = y.view(1, 1, 1, height, 1).expand(batch, 1, depth, height, width)
        x_axis = x_axis.view(1, 1, 1, 1, width).expand(
            batch, 1, depth, height, width
        )
        return torch.cat((z, y, x_axis), dim=1)

    def forward(self, w: torch.Tensor) -> dict[str, torch.Tensor]:
        features = (
            torch.cat((w, self.coordinate_channels(w)), dim=1)
            if self.use_coordinates
            else w
        )
        raw = self.unet(features)

        # Softplus avoids the hard 2.5 ceiling in the old implementation.
        fat = F.softplus(raw[:, 0:1], beta=1.0)
        delta_frequency_hz = self.max_residual_frequency_hz * torch.tanh(
            raw[:, 1:2]
        )
        log_gain = self.max_log_gain * torch.tanh(raw[:, 2:3])
        gain = torch.exp(log_gain)
        return {
            "fat": fat,
            "delta_frequency_hz": delta_frequency_hz,
            "gain": gain,
            "log_gain": log_gain,
        }


class DixonMagnitudeRenderer(nn.Module):
    """Two-echo single-peak Dixon magnitude renderer.

    ``fat_water_frequency_hz`` defaults to the frequency that produces exactly
    pi radians at TE_OP. Consequently TE_IP=2*TE_OP is nominally in phase.
    Protocol-specific measured values should replace these defaults when known.
    """

    def __init__(
        self,
        te_op_ms: float = 1.151,
        te_ip_ms: float = 2.302,
        t2_star_w_ms: float = 25.0,
        t2_star_f_ms: float = 20.0,
        fat_water_frequency_hz: float | None = None,
    ):
        super().__init__()
        if min(te_op_ms, te_ip_ms, t2_star_w_ms, t2_star_f_ms) <= 0:
            raise ValueError("Echo times and T2* constants must be positive")
        if fat_water_frequency_hz is None:
            fat_water_frequency_hz = 1.0 / (2.0 * te_op_ms * 1e-3)

        self.register_buffer("te_op_ms", torch.tensor(float(te_op_ms)))
        self.register_buffer("te_ip_ms", torch.tensor(float(te_ip_ms)))
        self.register_buffer("t2_star_w_ms", torch.tensor(float(t2_star_w_ms)))
        self.register_buffer("t2_star_f_ms", torch.tensor(float(t2_star_f_ms)))
        self.register_buffer(
            "fat_water_frequency_hz", torch.tensor(float(fat_water_frequency_hz))
        )

    def _echo(
        self,
        w: torch.Tensor,
        fat: torch.Tensor,
        delta_frequency_hz: torch.Tensor,
        gain: torch.Tensor,
        te_ms: torch.Tensor,
    ) -> torch.Tensor:
        w_echo = w * torch.exp(-te_ms / self.t2_star_w_ms)
        f_echo = fat * torch.exp(-te_ms / self.t2_star_f_ms)
        phase = (
            2.0
            * math.pi
            * (self.fat_water_frequency_hz + delta_frequency_hz)
            * te_ms
            * 1e-3
        )
        magnitude_squared = (
            w_echo.square()
            + f_echo.square()
            + 2.0 * w_echo * f_echo * torch.cos(phase)
        )
        return gain * torch.sqrt(magnitude_squared.clamp_min(1e-8))

    def forward(
        self,
        w: torch.Tensor,
        fat: torch.Tensor,
        delta_frequency_hz: torch.Tensor,
        gain: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "op": self._echo(w, fat, delta_frequency_hz, gain, self.te_op_ms),
            "ip": self._echo(w, fat, delta_frequency_hz, gain, self.te_ip_ms),
        }


class OfflineDixonDataset(Dataset):
    def __init__(
        self,
        pt_dataset_dir: str,
        patch_size: tuple[int, int, int] | None = None,
        foreground_patch_probability: float = 0.75,
    ):
        self.pt_files = sorted(Path(pt_dataset_dir).glob("*.pt"))
        if not self.pt_files:
            raise RuntimeError(f"No .pt files found in {pt_dataset_dir}")
        self.patch_size = tuple(patch_size) if patch_size is not None else None
        if self.patch_size is not None:
            if len(self.patch_size) != 3 or any(size <= 0 for size in self.patch_size):
                raise ValueError("patch_size must contain three positive integers")
            if any(size % 4 != 0 for size in self.patch_size):
                raise ValueError(
                    "Every patch dimension must be divisible by 4 for this U-Net"
                )
        if not 0.0 <= foreground_patch_probability <= 1.0:
            raise ValueError("foreground_patch_probability must be in [0, 1]")
        self.foreground_patch_probability = float(foreground_patch_probability)
    def __len__(self) -> int:
        return len(self.pt_files)

    def __getitem__(self, index: int):
        data = torch.load(self.pt_files[index], map_location="cpu", weights_only=True)
        required = ("W", "F", "IP", "OP", "mask")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{self.pt_files[index]} missing keys: {missing}")

        tensors = [data[key].float() for key in required[:-1]]
        mask = data["mask"].bool()
        all_tensors = tensors + [mask]
        if any(tensor.ndim != 4 for tensor in all_tensors):
            shapes = {key: tuple(data[key].shape) for key in required}
            raise ValueError(
                f"{self.pt_files[index]}: expected [C,D,H,W] tensors, got {shapes}"
            )
        spatial_shapes = {tuple(tensor.shape[-3:]) for tensor in all_tensors}
        if len(spatial_shapes) != 1:
            shapes = {key: tuple(data[key].shape) for key in required}
            raise ValueError(f"{self.pt_files[index]}: inconsistent shapes: {shapes}")

        if self.patch_size is not None:
            all_tensors = self._pad_and_crop(all_tensors)
        return tuple(all_tensors)

    def _pad_and_crop(self, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
        assert self.patch_size is not None
        patch_d, patch_h, patch_w = self.patch_size
        depth, height, width = tensors[0].shape[-3:]
        pad_d = max(0, patch_d - depth)
        pad_h = max(0, patch_h - height)
        pad_w = max(0, patch_w - width)
        if pad_d or pad_h or pad_w:
            padding = (0, pad_w, 0, pad_h, 0, pad_d)
            tensors = [F.pad(tensor, padding, value=0) for tensor in tensors]

        depth, height, width = tensors[0].shape[-3:]
        mask = tensors[-1]
        use_foreground = (
            mask.any()
            and torch.rand(()) < self.foreground_patch_probability
        )
        if use_foreground:
            foreground = torch.nonzero(mask[0], as_tuple=False)
            center = foreground[torch.randint(len(foreground), (1,)).item()]
            starts = [
                max(0, min(int(c) - size // 2, full - size))
                for c, size, full in zip(
                    center, self.patch_size, (depth, height, width)
                )
            ]
        else:
            starts = [
                torch.randint(full - size + 1, (1,)).item()
                for size, full in zip(self.patch_size, (depth, height, width))
            ]

        start_d, start_h, start_w = starts
        crop = (
            slice(None),
            slice(start_d, start_d + patch_d),
            slice(start_h, start_h + patch_h),
            slice(start_w, start_w + patch_w),
        )
        return [tensor[crop] for tensor in tensors]


def infinite_batches(loader: DataLoader):
    while True:
        yield from loader


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=value.dtype)
    channels = value.shape[1]
    denominator = mask.sum().clamp_min(1.0) * channels
    return (value * mask).sum() / denominator


def masked_charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    error = torch.sqrt((prediction - target).square() + epsilon**2) - epsilon
    return masked_mean(error, mask)


def masked_edge_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    total = prediction.new_zeros(())
    for dimension in (2, 3, 4):
        pred_diff = torch.diff(prediction, dim=dimension)
        target_diff = torch.diff(target, dim=dimension)
        mask_a = mask.narrow(dimension, 0, mask.shape[dimension] - 1)
        mask_b = mask.narrow(dimension, 1, mask.shape[dimension] - 1)
        pair_mask = mask_a & mask_b
        total = total + masked_charbonnier(pred_diff, target_diff, pair_mask)
    return total / 3.0


def edge_aware_tv(
    field: torch.Tensor,
    reference: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 5.0,
) -> torch.Tensor:
    total = field.new_zeros(())
    for dimension in (2, 3, 4):
        field_diff = torch.diff(field, dim=dimension).abs()
        image_diff = torch.diff(reference, dim=dimension).abs()
        weight = torch.exp(-alpha * image_diff)
        mask_a = mask.narrow(dimension, 0, mask.shape[dimension] - 1)
        mask_b = mask.narrow(dimension, 1, mask.shape[dimension] - 1)
        total = total + masked_mean(field_diff * weight, mask_a & mask_b)
    return total / 3.0


@dataclass
class TrainConfig:
    train_pt_dir: str
    val_pt_dir: str
    output_dir: str
    epochs: int = 300
    steps_per_epoch: int = 250
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    base_channels: int = 32
    use_coordinates: bool = False
    train_patch_size: tuple[int, int, int] = (128, 128, 64)
    foreground_patch_probability: float = 0.75
    num_workers: int = 4
    seed: int = 42
    lambda_edge: float = 0.10
    lambda_frequency_tv: float = 1e-4
    lambda_gain_tv: float = 1e-2
    lambda_gain_prior: float = 1e-2
    te_op_ms: float = 1.151
    te_ip_ms: float = 2.302
    t2_star_w_ms: float = 25.0
    t2_star_f_ms: float = 20.0
    resume: str | None = None


def compute_losses(
    model: PhysicsParameterNet,
    renderer: DixonMagnitudeRenderer,
    batch,
    device: torch.device,
    config: TrainConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    w, fat_target, ip_target, op_target, mask = (
        tensor.to(device, non_blocking=True) for tensor in batch
    )
    latents = model(w)
    rendered = renderer(
        w,
        latents["fat"],
        latents["delta_frequency_hz"],
        latents["gain"],
    )

    loss_f = masked_charbonnier(latents["fat"], fat_target, mask)
    loss_ip = masked_charbonnier(rendered["ip"], ip_target, mask)
    loss_op = masked_charbonnier(rendered["op"], op_target, mask)
    loss_edge = 0.5 * (
        masked_edge_loss(rendered["ip"], ip_target, mask)
        + masked_edge_loss(rendered["op"], op_target, mask)
    )
    frequency_tv = edge_aware_tv(
        latents["delta_frequency_hz"] / model.max_residual_frequency_hz,
        w,
        mask,
    )
    gain_tv = edge_aware_tv(latents["log_gain"], w, mask)
    gain_prior = masked_mean(latents["log_gain"].square(), mask)

    reconstruction = loss_f + loss_ip + loss_op
    total = (
        reconstruction
        + config.lambda_edge * loss_edge
        + config.lambda_frequency_tv * frequency_tv
        + config.lambda_gain_tv * gain_tv
        + config.lambda_gain_prior * gain_prior
    )
    metrics = {
        "total": total,
        "reconstruction": reconstruction,
        "f": loss_f,
        "ip": loss_ip,
        "op": loss_op,
        "edge": loss_edge,
        "frequency_tv": frequency_tv,
        "gain_tv": gain_tv,
        "gain_prior": gain_prior,
    }
    return total, metrics


def make_loader(
    directory: str,
    shuffle: bool,
    num_workers: int,
    patch_size: tuple[int, int, int] | None = None,
    foreground_patch_probability: float = 0.75,
) -> DataLoader:
    kwargs = {
        "dataset": OfflineDixonDataset(
            directory,
            patch_size=patch_size,
            foreground_patch_probability=foreground_patch_probability,
        ),
        "batch_size": 1,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def metric_accumulator() -> dict[str, float]:
    return {
        key: 0.0
        for key in (
            "total", "reconstruction", "f", "ip", "op", "edge",
            "frequency_tv", "gain_tv", "gain_prior",
        )
    }


def add_metrics(accumulator: dict[str, float], metrics: dict[str, torch.Tensor]) -> None:
    for key, value in metrics.items():
        accumulator[key] += float(value.detach().cpu())


def averaged(metrics: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in metrics.items()}


def train(config: TrainConfig) -> None:
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhysicsParameterNet(
        base_c=config.base_channels,
        use_coordinates=config.use_coordinates,
    ).to(device)
    renderer = DixonMagnitudeRenderer(
        te_op_ms=config.te_op_ms,
        te_ip_ms=config.te_ip_ms,
        t2_star_w_ms=config.t2_star_w_ms,
        t2_star_f_ms=config.t2_star_f_ms,
    ).to(device)
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01
    )

    start_epoch = 1
    best_val_reconstruction = float("inf")
    if config.resume:
        checkpoint_data = torch.load(config.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
        start_epoch = int(checkpoint_data["epoch"]) + 1
        best_val_reconstruction = float(checkpoint_data["best_val_reconstruction"])

    train_loader = make_loader(
        config.train_pt_dir,
        True,
        config.num_workers,
        patch_size=config.train_patch_size,
        foreground_patch_probability=config.foreground_patch_probability,
    )
    # Keep validation on complete volumes so model selection remains an exact
    # fixed-volume measurement, rather than a stochastic patch estimate.
    val_loader = make_loader(config.val_pt_dir, False, config.num_workers)
    train_iterator = infinite_batches(train_loader)

    print(
        f"Training patches: {config.train_patch_size}; "
        f"foreground probability={config.foreground_patch_probability:.2f}; "
        "validation=full volume"
    )

    use_amp = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp and amp_dtype == torch.float16
    )

    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_metrics = metric_accumulator()

        for _ in range(config.steps_per_epoch):
            batch = next(train_iterator)
            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
            )
            with amp_context:
                loss, metrics = compute_losses(model, renderer, batch, device, config)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            add_metrics(train_metrics, metrics)
            del loss, metrics, batch

        scheduler.step()
        train_metrics = averaged(train_metrics, config.steps_per_epoch)

        if use_amp:
            torch.cuda.empty_cache()

        model.eval()
        val_metrics = metric_accumulator()
        with torch.no_grad():
            for batch in val_loader:
                amp_context = (
                    torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
                )
                with amp_context:
                    val_loss, metrics = compute_losses(
                        model, renderer, batch, device, config
                    )
                add_metrics(val_metrics, metrics)
                del val_loss, metrics, batch
        val_metrics = averaged(val_metrics, len(val_loader))

        if use_amp:
            torch.cuda.empty_cache()

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch:03d}/{config.epochs} | {elapsed:.1f}s | "
            f"lr={optimizer.param_groups[0]['lr']:.3e} | "
            f"train_rec={train_metrics['reconstruction']:.5f} | "
            f"val_rec={val_metrics['reconstruction']:.5f}"
        )
        print(
            f"  val F={val_metrics['f']:.5f} "
            f"IP={val_metrics['ip']:.5f} OP={val_metrics['op']:.5f} "
            f"edge={val_metrics['edge']:.5f}"
        )

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_reconstruction": min(
                best_val_reconstruction, val_metrics["reconstruction"]
            ),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": asdict(config),
        }
        torch.save(checkpoint_data, output_dir / "checkpoint_latest.pth")

        # Selection uses fixed F+IP+OP reconstruction, never a learnable loss
        # weight, so checkpoint values remain comparable across epochs.
        if val_metrics["reconstruction"] < best_val_reconstruction:
            best_val_reconstruction = val_metrics["reconstruction"]
            checkpoint_data["best_val_reconstruction"] = best_val_reconstruction
            torch.save(checkpoint_data, output_dir / "checkpoint_best.pth")
            print(f"  New best fixed reconstruction loss: {best_val_reconstruction:.6f}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pt-dir", required=True)
    parser.add_argument("--val-pt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--steps-per-epoch", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument(
        "--use-coordinates",
        action="store_true",
        help="Enable crop-relative xyz channels; disabled by default for cross-dataset transfer",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--train-patch-size",
        type=int,
        nargs=3,
        metavar=("D", "H", "W"),
        default=(128, 128, 64),
        help="Fixed training patch size; each value must be divisible by 4",
    )
    parser.add_argument(
        "--foreground-patch-probability",
        type=float,
        default=0.75,
        help="Probability that a training patch is centered on the valid mask",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-edge", type=float, default=0.10)
    parser.add_argument("--lambda-frequency-tv", type=float, default=1e-4)
    parser.add_argument("--lambda-gain-tv", type=float, default=1e-2)
    parser.add_argument("--lambda-gain-prior", type=float, default=1e-2)
    parser.add_argument("--te-op-ms", type=float, default=1.151)
    parser.add_argument("--te-ip-ms", type=float, default=2.302)
    parser.add_argument("--t2-star-w-ms", type=float, default=25.0)
    parser.add_argument("--t2-star-f-ms", type=float, default=20.0)
    parser.add_argument("--resume", default=None)
    arguments = vars(parser.parse_args())
    arguments["train_patch_size"] = tuple(arguments["train_patch_size"])
    return TrainConfig(**arguments)


if __name__ == "__main__":
    train(parse_args())
