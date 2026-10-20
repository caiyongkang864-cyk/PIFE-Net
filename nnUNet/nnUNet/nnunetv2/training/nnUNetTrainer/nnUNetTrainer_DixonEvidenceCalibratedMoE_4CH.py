"""Independent W/IP/OP nnU-Net experts with evidence-calibrated soft anchoring.

Input order is fixed:
    channel 0: original W / T1w-FS
    channel 1: synthetic F
    channel 2: synthetic IP
    channel 3: synthetic OP

The architecture deliberately preserves the two mechanisms supported by the
existing experiments:

1. The successful PGAN model used three complete, independent segmentation
   networks and late voxel-wise logit fusion.
2. The Anchored model protected the original W prediction from unreliable
   synthetic modalities.

W, IP and OP are fed directly to three independent complete nnU-Nets. F is
deliberately excluded from both segmentation and routing because prior
inference-level diagnostics did not show a positive F contribution. IP/OP
physical evidence, detached expert confidence and case-by-class quality are
used to construct normalized W/IP/OP late-fusion weights.

This revision removes the previous triple OP bias. OP receives a modest prior
only in the base reliability term; class strength and confidence eligibility
start symmetrically for IP and OP. Confidence is a bounded eligibility signal,
not an unbounded positive boost. A differentiable low-W-weight penalty gives W
a real soft protection: auxiliary experts may still dominate, but only when
the fused segmentation gain is large enough to pay that optimization cost.

Important: ground truth is used only during validation to measure whether
confidence gain predicts real repair or harm. Those diagnostics never enter the
training loss and therefore do not leak labels into the router.

Safe efficiency changes only:
    * W standardization and W gradients are computed once;
    * each physical modality is standardized, differentiated and projected once;
    * one full-resolution router is shared by all deep-supervision outputs;
    * no expert encoder or decoder is removed or replaced by a shallow head.

The fused loss is augmented with direct W/IP/OP expert losses. This prevents
the auxiliary experts from receiving almost no gradient when the W anchor is
dominant. The inference output remains the fused nnU-Net output only.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import autocast, nn
from torch import distributed as dist
import torch.nn.functional as F
from torch._dynamo import OptimizedModule

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context


class DifferentialPolyLRScheduler:
    """PolyLR that preserves a fixed LR ratio between parameter groups."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_lrs: Sequence[float],
        max_epochs: int,
        exponent: float = 0.9,
    ) -> None:
        if len(optimizer.param_groups) != len(initial_lrs):
            raise ValueError("one initial LR is required per parameter group")
        if int(max_epochs) <= 0:
            raise ValueError("max_epochs must be positive")
        self.optimizer = optimizer
        self.initial_lrs = tuple(float(value) for value in initial_lrs)
        self.max_epochs = int(max_epochs)
        self.exponent = float(exponent)
        self.current_epoch = 0
        self._set_lrs(0)

    def _factor(self, epoch: int) -> float:
        remaining = max(0.0, 1.0 - float(epoch) / float(self.max_epochs))
        return remaining**self.exponent

    def _set_lrs(self, epoch: int) -> None:
        factor = self._factor(epoch)
        for group, initial_lr in zip(self.optimizer.param_groups, self.initial_lrs):
            group["lr"] = initial_lr * factor

    def step(self, current_epoch: int | None = None) -> None:
        if current_epoch is None:
            current_epoch = self.current_epoch + 1
        self.current_epoch = int(current_epoch)
        self._set_lrs(self.current_epoch)

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, object]:
        return {
            "initial_lrs": self.initial_lrs,
            "max_epochs": self.max_epochs,
            "exponent": self.exponent,
            "current_epoch": self.current_epoch,
        }

    def load_state_dict(self, state_dict: Mapping[str, object]) -> None:
        initial_lrs = tuple(float(value) for value in state_dict["initial_lrs"])
        if len(initial_lrs) != len(self.optimizer.param_groups):
            raise RuntimeError("scheduler parameter-group count changed")
        self.initial_lrs = initial_lrs
        self.max_epochs = int(state_dict["max_epochs"])
        self.exponent = float(state_dict["exponent"])
        self.current_epoch = int(state_dict["current_epoch"])
        self._set_lrs(self.current_epoch)


def _valid_group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _resize(image: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    if tuple(image.shape[2:]) == tuple(size):
        return image
    return F.interpolate(image, size=size, mode="trilinear", align_corners=False)


def _spatial_standardize(image: torch.Tensor) -> torch.Tensor:
    """Per-sample standardization without mixing patients in a mini-batch."""

    spatial_dimensions = tuple(range(2, image.ndim))
    work = image.float()
    mean = work.mean(dim=spatial_dimensions, keepdim=True)
    variance = (work - mean).square().mean(
        dim=spatial_dimensions, keepdim=True
    )
    normalized = (work - mean) / torch.sqrt(variance + 1e-5)
    return normalized.clamp_(-6.0, 6.0).to(image.dtype)


def _gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    dz = F.pad(image[:, :, 1:] - image[:, :, :-1], (0, 0, 0, 0, 0, 1))
    dy = F.pad(image[:, :, :, 1:] - image[:, :, :, :-1], (0, 0, 0, 1, 0, 0))
    dx = F.pad(image[:, :, :, :, 1:] - image[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))
    return torch.sqrt(dx.square() + dy.square() + dz.square() + 1e-6)


def _logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-5), 1.0 - 1e-5)
    return math.log(probability / (1.0 - probability))


class PhysicalMaskProjector(nn.Module):
    """Compress one Dixon descriptor into a router-visible physical map."""

    def __init__(self, hidden_channels: int = 16) -> None:
        super().__init__()
        groups = _valid_group_count(hidden_channels)
        self.net = nn.Sequential(
            nn.Conv3d(4, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, 1, 1, bias=True),
        )
        # Start as a neutral router feature. The projector receives gradients
        # from the fused segmentation objective through the reliability router.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(descriptor))


class IndependentReliabilityRouter(nn.Module):
    """Factorized IP/OP reliability over case, class and space.

    The case-by-class and local-spatial factors are deliberately returned
    separately. Expanding [B, 2, C, D, H, W] here would create a very large
    full-resolution tensor (252 maps for B=2 and C=63) before fusion.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        initial_reliability: Sequence[float],
    ) -> None:
        super().__init__()
        if len(initial_reliability) != 2:
            raise ValueError("expected initial reliability for IP and OP")
        self.num_classes = int(num_classes)
        hidden_channels = 24
        self.local_path = nn.Sequential(
            nn.Conv3d(input_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(
                _valid_group_count(hidden_channels), hidden_channels
            ),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, 2, 1, bias=False),
        )
        self.case_path = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(input_channels, hidden_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, 2 * self.num_classes, 1),
        )
        nn.init.zeros_(self.local_path[-1].weight)
        nn.init.zeros_(self.case_path[-1].weight)
        nn.init.zeros_(self.case_path[-1].bias)

        base = torch.empty(2, self.num_classes)
        base[0].fill_(_logit(initial_reliability[0]))
        base[1].fill_(_logit(initial_reliability[1]))
        self.register_buffer(
            "base_reliability_prior", base.clone(), persistent=True
        )
        self.base_reliability_logits = nn.Parameter(base.clone())

        self.last_case_class_reliability: torch.Tensor | None = None

    def forward(
        self,
        evidence: torch.Tensor,
        availability: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = evidence.shape[0]
        local_logits = self.local_path(evidence)
        case_class_logits = self.case_path(evidence).reshape(
            batch_size, 2, self.num_classes, 1, 1, 1
        )
        # The learned base term is a bounded correction around the fixed
        # prior. This prevents a global modality preference from growing
        # without limit and bypassing local/case evidence.
        bounded_base = self.base_reliability_prior + 0.5 * torch.tanh(
            self.base_reliability_logits - self.base_reliability_prior
        )
        base_logits = bounded_base.reshape(
            1, 2, self.num_classes, 1, 1, 1
        )

        availability_maps = torch.stack(
            [availability["ip"], availability["op"]], dim=1
        ).reshape(batch_size, 2, 1, 1, 1)
        local_logits = torch.where(
            availability_maps,
            local_logits,
            torch.full_like(local_logits, -20.0),
        )
        self.last_case_class_reliability = (
            torch.sigmoid(base_logits + case_class_logits).detach()
            if not self.training
            else None
        )
        return local_logits, case_class_logits, base_logits


class DixonEvidenceCalibratedMoE4CH(nn.Module):
    """Three full experts with OP-first evidence and breakable W anchoring."""

    modality_names = ("ip", "op")

    def __init__(
        self,
        unet_w: nn.Module,
        unet_ip: nn.Module,
        unet_op: nn.Module,
        num_classes: int,
        modality_dropout: Mapping[str, float],
        confidence_thresholds: Sequence[float] = (0.20, 0.20),
        confidence_temperatures: Sequence[float] = (0.12, 0.12),
        initial_w_anchor_logit: float = 2.0,
        minimum_w_weight: float = 0.55,
        local_score_limit: float = 2.0,
        case_score_limit: float = 1.5,
        class_score_limit: float = 0.75,
        record_fusion_maps: bool = False,
    ) -> None:
        super().__init__()
        self.unet_w = unet_w
        self.unet_ip = unet_ip
        self.unet_op = unet_op
        self.projectors = nn.ModuleDict(
            {name: PhysicalMaskProjector() for name in self.modality_names}
        )
        self.modality_dropout = {
            name: float(modality_dropout[name]) for name in self.modality_names
        }

        if len(confidence_thresholds) != 2:
            raise ValueError(
                "confidence_thresholds must contain [IP, OP]"
            )
        
        if len(confidence_temperatures) != 2:
            raise ValueError(
                "confidence_temperatures must contain [IP, OP]"
            )
        
        self.confidence_thresholds = {
            "ip": float(confidence_thresholds[0]),
            "op": float(confidence_thresholds[1]),
        }
        
        self.confidence_temperatures = {
            "ip": float(confidence_temperatures[0]),
            "op": float(confidence_temperatures[1]),
        }
        
        for name in ("ip", "op"):
            if self.confidence_temperatures[name] <= 0.0:
                raise ValueError(
                    f"{name} confidence temperature must be positive"
                )
        
        self.record_fusion_maps = bool(record_fusion_maps)
        self.minimum_w_weight = float(minimum_w_weight)
        self.local_score_limit = float(local_score_limit)
        self.case_score_limit = float(case_score_limit)
        self.class_score_limit = float(class_score_limit)
        if not 0.0 < self.minimum_w_weight < 1.0:
            raise ValueError("minimum_w_weight must be in (0, 1)")
        for name, value in (
            ("local_score_limit", self.local_score_limit),
            ("case_score_limit", self.case_score_limit),
            ("class_score_limit", self.class_score_limit),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        

        # W + 2 masks + 2 differences + 2 edge agreements + W confidence +
        # IP/OP confidence gains = 10 channels. F is intentionally absent.
        self.router = IndependentReliabilityRouter(
            input_channels=10,
            num_classes=num_classes,
            # The only hand-written OP preference in the complete router.
            initial_reliability=(0.25, 0.35),
        )

        output_count = len(self.unet_w.decoder.seg_layers)
        # Symmetric, neutral IP/OP class corrections. Modality/class
        # preferences must now be learned from segmentation evidence instead
        # of being duplicated in initialization.
        strengths = torch.zeros(output_count, 2, int(num_classes))
        self.class_strength_logits = nn.Parameter(strengths)
        w_anchor_base = torch.full(
            (output_count, int(num_classes)), float(initial_w_anchor_logit)
        )
        if int(num_classes) > 0:
            w_anchor_base[:, 0].fill_(3.0)
        self.register_buffer("w_anchor_base", w_anchor_base, persistent=True)
        self.w_anchor_delta = nn.Parameter(torch.zeros_like(w_anchor_base))

        self.last_confidence_gates: dict[str, torch.Tensor] | None = None
        self.last_confidence_gains: dict[str, torch.Tensor] | None = None
        self.last_effective_weights: dict[str, torch.Tensor] | None = None

    def _availability(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = reference.shape[0]
        result = {}
        for name in self.modality_names:
            drop_probability = self.modality_dropout[name]
            if self.training and drop_probability > 0.0:
                result[name] = (
                    torch.rand(batch_size, device=reference.device)
                    >= drop_probability
                )
            else:
                result[name] = torch.ones(
                    batch_size, dtype=torch.bool, device=reference.device
                )
        return result

    @staticmethod
    def _availability_map(
        availability: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        spatial_ones = (1,) * (reference.ndim - 2)
        return availability.reshape(
            reference.shape[0], 1, *spatial_ones
        ).to(reference.dtype)

    def _physical_features(
        self,
        w: torch.Tensor,
        auxiliary: Mapping[str, torch.Tensor],
        availability: Mapping[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        # These W operations were repeated by each modality in earlier code.
        w_standard = _spatial_standardize(w)
        w_edge = _gradient_magnitude(w_standard)
        masks = {}
        differences = {}
        edge_agreements = {}

        for name in self.modality_names:
            aux_standard = _spatial_standardize(auxiliary[name])
            aux_edge = _gradient_magnitude(aux_standard)
            difference = torch.abs(aux_standard - w_standard)
            edge_agreement = torch.exp(-torch.abs(aux_edge - w_edge))
            descriptor = torch.cat(
                (aux_standard, difference, aux_edge, w_edge), dim=1
            )
            available = self._availability_map(
                availability[name], descriptor
            )
            masks[name] = self.projectors[name](descriptor) * available
            differences[name] = difference * available
            edge_agreements[name] = edge_agreement * available

        return w_standard, masks, differences, edge_agreements

    @staticmethod
    def _highest_output(outputs):
        return outputs[0] if isinstance(outputs, (list, tuple)) else outputs

    @staticmethod
    def _confidence(logits: torch.Tensor) -> torch.Tensor:
        # Routing observations are detached to prevent confidence inflation.
        with torch.no_grad():
            return torch.softmax(logits.detach().float(), dim=1).amax(
                dim=1, keepdim=True
            ).to(logits.dtype)

    def _router_evidence(
        self,
        w_standard: torch.Tensor,
        masks: Mapping[str, torch.Tensor],
        differences: Mapping[str, torch.Tensor],
        edge_agreements: Mapping[str, torch.Tensor],
        w_logits: torch.Tensor,
        ip_logits: torch.Tensor,
        op_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output_size = w_logits.shape[2:]
        confidence_w = self._confidence(w_logits)
        confidence_gains = {
            "ip": self._confidence(ip_logits) - confidence_w,
            "op": self._confidence(op_logits) - confidence_w,
        }
        self.last_confidence_gains = (
            {name: value.detach() for name, value in confidence_gains.items()}
            if not self.training
            else None
        )
        evidence_parts = [
            _resize(w_standard, output_size),
            *[
                _resize(masks[name], output_size)
                for name in self.modality_names
            ],
            *[
                _resize(differences[name], output_size)
                for name in self.modality_names
            ],
            *[
                _resize(edge_agreements[name], output_size)
                for name in self.modality_names
            ],
            confidence_w,
            confidence_gains["ip"],
            confidence_gains["op"],
        ]
        return torch.cat(evidence_parts, dim=1), confidence_gains

    def _blend_one_scale(
        self,
        w_logits: torch.Tensor,
        ip_logits: torch.Tensor,
        op_logits: torch.Tensor,
        reliability_factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        high_resolution_gains: Mapping[str, torch.Tensor],
        output_index: int,
    ) -> torch.Tensor:
        output_size = w_logits.shape[2:]
        local_logits, case_class_logits, base_logits = reliability_factors
        local_logits = _resize(local_logits, output_size).to(w_logits.dtype)
        case_class_logits = case_class_logits.to(w_logits.dtype)
        base_logits = base_logits.to(w_logits.dtype)

        if tuple(high_resolution_gains["ip"].shape[2:]) == tuple(output_size):
            confidence_gains = high_resolution_gains
        else:
            confidence_w = self._confidence(w_logits)
            confidence_gains = {
                "ip": self._confidence(ip_logits) - confidence_w,
                "op": self._confidence(op_logits) - confidence_w,
            }

        confidence_gates = {
            name: torch.sigmoid(
                (
                    confidence_gains[name]
                    - self.confidence_thresholds[name]
                ) / self.confidence_temperatures[name]
            )
            for name in ("ip", "op")
        }
        # log(gate) is always <= 0. Confidence can reject an auxiliary expert
        # but can no longer create an arbitrarily large positive score merely
        # because that expert is over-confident. Actual breakthrough requires
        # consistent local and case-by-class reliability evidence as well.
        confidence_scores = {
            name: torch.log(gate.clamp_min(1e-4))
            for name, gate in confidence_gates.items()
        }

        spatial_ones = (1,) * (w_logits.ndim - 2)
        class_scores = (
            self.class_score_limit
            * torch.tanh(
                self.class_strength_logits[output_index]
                / self.class_score_limit
            )
        ).to(w_logits.dtype).reshape(1, 2, -1, *spatial_ones)
        # Keep a bounded positive W prior instead of a brittle hard minimum.
        # Strong auxiliary evidence can still exceed this score in the softmax.
        w_score = (
            self.w_anchor_base[output_index]
            + 0.5 * torch.tanh(self.w_anchor_delta[output_index])
        ).to(w_logits.dtype).reshape(1, 1, -1, *spatial_ones)
        bounded_local = self.local_score_limit * torch.tanh(
            local_logits / self.local_score_limit
        )
        bounded_case = self.case_score_limit * torch.tanh(
            case_class_logits / self.case_score_limit
        )
        ip_score = (
            bounded_local[:, 0:1].unsqueeze(2)
            + bounded_case[:, 0:1]
            + base_logits[:, 0:1]
            + class_scores[:, 0:1]
            + confidence_scores["ip"].unsqueeze(1)
        ).to(w_logits.dtype)
        op_score = (
            bounded_local[:, 1:2].unsqueeze(2)
            + bounded_case[:, 1:2]
            + base_logits[:, 1:2]
            + class_scores[:, 1:2]
            + confidence_scores["op"].unsqueeze(1)
        ).to(w_logits.dtype)

        # Exact three-way softmax expressed relative to the W score. This
        # avoids concatenating [B, 3, C, D, H, W] and prevents Inductor from
        # allocating the 252-map FP32 temporary seen in the original version.
        ip_relative = torch.exp(
            torch.clamp(ip_score - w_score, min=-12.0, max=12.0)
        )[:, 0]
        op_relative = torch.exp(
            torch.clamp(op_score - w_score, min=-12.0, max=12.0)
        )[:, 0]
        denominator = 1.0 + ip_relative + op_relative
        w_weight = denominator.reciprocal()
        ip_weight = ip_relative * w_weight
        op_weight = op_relative * w_weight

        # This is still late logit fusion. Unlike the old 0.45 hard cap, the
        # learned W prior can be exceeded when reliability and confidence
        # evidence are sufficiently strong.
        fused = (
            w_weight * w_logits
            + ip_weight * ip_logits
            + op_weight * op_logits
        )

        # A true soft W protection. This is not a hard clamp: W may fall below
        # the target when doing so improves the supervised fused loss enough,
        # but low-W solutions now have an explicit differentiable cost.
        w_anchor_penalty = F.relu(
            self.minimum_w_weight - w_weight
        ).square().mean()
        fusion_statistics = torch.stack(
            (
                w_anchor_penalty,
                w_weight.mean(),
                ip_weight.mean(),
                op_weight.mean(),
            )
        )

        if not self.training and output_index == 0 and self.record_fusion_maps:
            self.last_confidence_gates = {
                name: value.detach()
                for name, value in confidence_gates.items()
            }
            self.last_effective_weights = {
                "w": w_weight.detach(),
                "ip": ip_weight.detach(),
                "op": op_weight.detach(),
            }
        return fused, fusion_statistics

    def forward(self, x: torch.Tensor, return_experts: bool = False):
        if x.ndim != 5 or x.shape[1] != 4:
            raise ValueError(
                "expected [B,4,D,H,W] ordered [W,F,IP,OP], "
                f"got {tuple(x.shape)}"
            )
        w = x[:, 0:1]
        auxiliary = {
            "fat": x[:, 1:2],
            "ip": x[:, 2:3],
            "op": x[:, 3:4],
        }
        availability = self._availability(w)
        w_standard, masks, differences, edge_agreements = (
            self._physical_features(w, auxiliary, availability)
        )

        # Each expert receives its own sequence directly. This is the critical
        # difference from W * (1 + mask): IP/OP keep their complete Dixon
        # contrast instead of becoming slightly perturbed W images.
        w_outputs = self.unet_w(w)
        ip_outputs = self.unet_ip(auxiliary["ip"])
        op_outputs = self.unet_op(auxiliary["op"])
        w_high = self._highest_output(w_outputs)
        ip_high = self._highest_output(ip_outputs)
        op_high = self._highest_output(op_outputs)

        router_evidence, confidence_gains = self._router_evidence(
            w_standard,
            masks,
            differences,
            edge_agreements,
            w_high,
            ip_high,
            op_high,
        )
        reliability_factors = self.router(router_evidence, availability)

        if isinstance(w_outputs, (list, tuple)):
            blended_outputs = [
                self._blend_one_scale(
                    w_output,
                    ip_output,
                    op_output,
                    reliability_factors,
                    confidence_gains,
                    output_index,
                )
                for output_index, (w_output, ip_output, op_output) in enumerate(
                    zip(w_outputs, ip_outputs, op_outputs)
                )
            ]
            fused_outputs = [item[0] for item in blended_outputs]
            # The highest-resolution output carries the clinically meaningful
            # router geometry and is used for the soft-anchor regularizer.
            fusion_statistics = blended_outputs[0][1]
        else:
            fused_outputs, fusion_statistics = self._blend_one_scale(
                w_outputs,
                ip_outputs,
                op_outputs,
                reliability_factors,
                confidence_gains,
                0,
            )

        if return_experts:
            return (
                fused_outputs,
                w_outputs,
                ip_outputs,
                op_outputs,
                fusion_statistics,
            )
        return fused_outputs

    def compute_conv_feature_map_size(self, input_size):
        # nnU-Net uses this for architecture inspection. The physical projector
        # and router are small relative to the three complete segmentation nets.
        return (
            self.unet_w.compute_conv_feature_map_size(input_size)
            + self.unet_ip.compute_conv_feature_map_size(input_size)
            + self.unet_op.compute_conv_feature_map_size(input_size)
        )


class nnUNetTrainer_DixonEvidenceCalibratedMoE_4CH(nnUNetTrainer):
    """Independent experts, OP-first soft anchor and confidence diagnostics."""

    _VALID_TRAINING_MODES = {
        "scratch",
        "pretrained",
        "independent_pretrained",
    }
    _CONFIDENCE_BIN_EDGES = (-float("inf"), -0.20, -0.10, 0.0, 0.10, 0.20, float("inf"))

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        # nnUNetTrainer introspects these exact argument names. Do not replace
        # this signature by *args/**kwargs.
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )

        mode = os.environ.get("DIXON_TRAINING_MODE", "scratch").strip().lower()
        if mode not in self._VALID_TRAINING_MODES:
            raise ValueError(
                "DIXON_TRAINING_MODE must be scratch, pretrained, or "
                "independent_pretrained"
            )
        checkpoint = os.environ.get("DIXON_PRETRAINED_WEIGHTS", "").strip()
        if mode != "scratch" and not checkpoint:
            raise ValueError(
                "DIXON_PRETRAINED_WEIGHTS is required outside scratch mode"
            )
        self.dixon_training_mode = mode
        self.dixon_pretrained_weights = checkpoint or None

        self.dixon_w_lr_multiplier = self._read_multiplier(
            "DIXON_W_LR_MULTIPLIER", 0.20
        )
        self.dixon_expert_lr_multiplier = self._read_multiplier(
            "DIXON_EXPERT_LR_MULTIPLIER", 0.50
        )
        self.dixon_fusion_lr_multiplier = self._read_multiplier(
            "DIXON_FUSION_LR_MULTIPLIER", 1.00
        )
        self.w_expert_loss_weight = self._read_nonnegative(
            "DIXON_W_EXPERT_LOSS_WEIGHT", 0.30
        )
        self.ip_expert_loss_weight = self._read_nonnegative(
            "DIXON_IP_EXPERT_LOSS_WEIGHT", 0.15
        )
        self.op_expert_loss_weight = self._read_nonnegative(
            "DIXON_OP_EXPERT_LOSS_WEIGHT", 0.15
        )
        self.w_anchor_penalty_weight = self._read_nonnegative(
            "DIXON_W_ANCHOR_PENALTY_WEIGHT", 0.25
        )

        epoch_text = os.environ.get("DIXON_NUM_EPOCHS", "").strip()
        if epoch_text:
            try:
                requested_epochs = int(epoch_text)
            except ValueError as error:
                raise ValueError("DIXON_NUM_EPOCHS must be an integer") from error
            if requested_epochs <= 0:
                raise ValueError("DIXON_NUM_EPOCHS must be positive")
            self.num_epochs = requested_epochs

    @staticmethod
    def _read_nonnegative(name: str, default: float) -> float:
        text = os.environ.get(name, str(default)).strip()
        try:
            value = float(text)
        except ValueError as error:
            raise ValueError(f"{name} must be a floating-point number") from error
        if value < 0.0:
            raise ValueError(f"{name} must be nonnegative")
        return value

    @staticmethod
    def _read_multiplier(name: str, default: float) -> float:
        value = nnUNetTrainer_DixonEvidenceCalibratedMoE_4CH._read_nonnegative(
            name, default
        )
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
        return value

    def _unwrapped_network(self) -> nn.Module:
        network = getattr(self.network, "module", self.network)
        if isinstance(network, OptimizedModule):
            network = network._orig_mod
        return network

    @staticmethod
    def _canonical_checkpoint_key(key: str) -> str:
        prefixes = ("module.", "_orig_mod.", "network.")
        canonical = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if canonical.startswith(prefix):
                    canonical = canonical[len(prefix) :]
                    changed = True
        return canonical

    def _load_w_baseline_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"W-baseline checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError(f"checkpoint is not a mapping: {path}")
        if "network_weights" in checkpoint:
            raw_state = checkpoint["network_weights"]
        elif "state_dict" in checkpoint:
            raw_state = checkpoint["state_dict"]
        else:
            raw_state = checkpoint
        if not isinstance(raw_state, Mapping):
            raise RuntimeError("checkpoint has no valid network state mapping")

        source_state = {
            self._canonical_checkpoint_key(key): value
            for key, value in raw_state.items()
            if isinstance(key, str) and torch.is_tensor(value)
        }
        network = self._unwrapped_network()
        expected_state = network.unet_w.state_dict()
        missing = []
        mismatched = []
        loaded_state = {}
        for key, target in expected_state.items():
            source = source_state.get(key)
            if source is None:
                missing.append(key)
            elif tuple(source.shape) != tuple(target.shape):
                mismatched.append(
                    f"{key}: source={tuple(source.shape)}, "
                    f"target={tuple(target.shape)}"
                )
            else:
                loaded_state[key] = source
        if missing or mismatched:
            raise RuntimeError(
                "W checkpoint is incompatible. Missing (first 10): "
                + ", ".join(missing[:10])
                + " | mismatched (first 10): "
                + "; ".join(mismatched[:10])
            )

        # All experts begin from the same proven W solution. Their independent
        # parameters, direct sequence inputs and losses then permit specialization.
        network.unet_w.load_state_dict(loaded_state, strict=True)
        network.unet_ip.load_state_dict(loaded_state, strict=True)
        network.unet_op.load_state_dict(loaded_state, strict=True)
        self.print_to_log_file(
            f"Loaded {len(loaded_state)} tensors from W baseline {path} "
            "into all three complete experts."
        )

    def _load_independent_experts(self, checkpoint_path: str) -> None:
        """Load only mature W/IP/OP experts from an old Independent model."""
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Independent checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "network_weights" in checkpoint:
            raw_state = checkpoint["network_weights"]
        elif "state_dict" in checkpoint:
            raw_state = checkpoint["state_dict"]
        else:
            raw_state = checkpoint
        if not isinstance(raw_state, Mapping):
            raise RuntimeError("checkpoint has no valid network state mapping")

        source_state = {
            self._canonical_checkpoint_key(key): value
            for key, value in raw_state.items()
            if isinstance(key, str) and torch.is_tensor(value)
        }
        network = self._unwrapped_network()
        for expert_name in ("unet_w", "unet_ip", "unet_op"):
            expert = getattr(network, expert_name)
            expected = expert.state_dict()
            prefix = expert_name + "."
            selected = {
                key: source_state[prefix + key]
                for key in expected
                if prefix + key in source_state
                and tuple(source_state[prefix + key].shape)
                == tuple(expected[key].shape)
            }
            missing = sorted(set(expected) - set(selected))
            if missing:
                raise RuntimeError(
                    f"{expert_name} checkpoint mismatch; missing first 10: "
                    + ", ".join(missing[:10])
                )
            expert.load_state_dict(selected, strict=True)
        self.print_to_log_file(
            f"Loaded mature W/IP/OP experts from {path}; new router and soft "
            "anchor remain freshly initialized."
        )

    def initialize(self) -> None:
        super().initialize()
        if self.dixon_training_mode == "pretrained":
            self._load_w_baseline_checkpoint(self.dixon_pretrained_weights)
        elif self.dixon_training_mode == "independent_pretrained":
            self._load_independent_experts(self.dixon_pretrained_weights)
        else:
            self.print_to_log_file(
                "Dixon mode: scratch. W/IP/OP use independent full-network "
                "initializations and direct expert supervision."
            )

    def configure_optimizers(self):
        if self.dixon_training_mode == "scratch":
            return super().configure_optimizers()

        network = self._unwrapped_network()
        w_parameters = list(network.unet_w.parameters())
        expert_parameters = list(network.unet_ip.parameters()) + list(
            network.unet_op.parameters()
        )
        fusion_parameters = (
            list(network.projectors.parameters())
            + list(network.router.parameters())
            + [network.class_strength_logits, network.w_anchor_delta]
        )
        if not w_parameters or not expert_parameters or not fusion_parameters:
            raise RuntimeError("failed to construct optimizer parameter groups")

        w_lr = self.initial_lr * self.dixon_w_lr_multiplier
        expert_lr = self.initial_lr * self.dixon_expert_lr_multiplier
        fusion_lr = self.initial_lr * self.dixon_fusion_lr_multiplier
        optimizer = torch.optim.SGD(
            [
                {
                    "params": w_parameters,
                    "lr": w_lr,
                    "group_name": "w_expert",
                },
                {
                    "params": expert_parameters,
                    "lr": expert_lr,
                    "group_name": "ip_op_experts",
                },
                {
                    "params": fusion_parameters,
                    "lr": fusion_lr,
                    "group_name": "physical_fusion",
                },
            ],
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True,
        )
        scheduler = DifferentialPolyLRScheduler(
            optimizer,
            (w_lr, expert_lr, fusion_lr),
            self.num_epochs,
            exponent=0.9,
        )
        return optimizer, scheduler

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [
                item.to(self.device, non_blocking=True) for item in target
            ]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        context = (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )
        with context:
            (
                fused,
                w_outputs,
                ip_outputs,
                op_outputs,
                fusion_statistics,
            ) = self.network(
                data, return_experts=True
            )
            fused_loss = self.loss(fused, target)
            w_loss = self.loss(w_outputs, target)
            ip_loss = self.loss(ip_outputs, target)
            op_loss = self.loss(op_outputs, target)
            optimization_loss = (
                fused_loss
                + self.w_expert_loss_weight * w_loss
                + self.ip_expert_loss_weight * ip_loss
                + self.op_expert_loss_weight * op_loss
                + self.w_anchor_penalty_weight * fusion_statistics[0]
            )

        if self.grad_scaler is not None:
            self.grad_scaler.scale(optimization_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            optimization_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        # Keep the standard nnU-Net train_loss directly comparable: it is the
        # fused-output loss, not the weighted sum of auxiliary objectives.
        return {
            "loss": fused_loss.detach().cpu().numpy(),
            "optimization_loss": optimization_loss.detach().cpu().numpy(),
            "w_loss": w_loss.detach().cpu().numpy(),
            "ip_loss": ip_loss.detach().cpu().numpy(),
            "op_loss": op_loss.detach().cpu().numpy(),
            "w_anchor_penalty": fusion_statistics[0].detach().cpu().numpy(),
            "mean_w_weight": fusion_statistics[1].detach().cpu().numpy(),
            "mean_ip_weight": fusion_statistics[2].detach().cpu().numpy(),
            "mean_op_weight": fusion_statistics[3].detach().cpu().numpy(),
        }

    @classmethod
    def _confidence_repair_counts(
        cls,
        w_logits: torch.Tensor,
        auxiliary_logits: torch.Tensor,
        confidence_gain: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> np.ndarray:
        """Count disagreement/repair/harm by confidence-gain bin.

        Columns are: disagreement, repair, harm, neither_correct. These are
        validation diagnostics only; returned NumPy counts have no gradient.
        """
        w_prediction = w_logits.argmax(1)
        auxiliary_prediction = auxiliary_logits.argmax(1)
        target_map = target[:, 0].long()
        disagreement = w_prediction != auxiliary_prediction
        if valid_mask is not None:
            disagreement = disagreement & valid_mask[:, 0].bool()

        w_correct = w_prediction == target_map
        auxiliary_correct = auxiliary_prediction == target_map
        repair = disagreement & (~w_correct) & auxiliary_correct
        harm = disagreement & w_correct & (~auxiliary_correct)
        neither = disagreement & (~w_correct) & (~auxiliary_correct)
        delta = confidence_gain[:, 0]

        result = torch.zeros(
            len(cls._CONFIDENCE_BIN_EDGES) - 1,
            4,
            dtype=torch.int64,
            device=w_logits.device,
        )
        for index, (low, high) in enumerate(
            zip(cls._CONFIDENCE_BIN_EDGES[:-1], cls._CONFIDENCE_BIN_EDGES[1:])
        ):
            in_bin = disagreement & (delta >= low) & (delta < high)
            result[index, 0] = in_bin.sum()
            result[index, 1] = (repair & in_bin).sum()
            result[index, 2] = (harm & in_bin).sum()
            result[index, 3] = (neither & in_bin).sum()
        return result.cpu().numpy()

    def validation_step(self, batch: dict) -> dict:
        """Standard nnU-Net online validation plus confidence calibration."""
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [item.to(self.device, non_blocking=True) for item in target]
        else:
            target = target.to(self.device, non_blocking=True)

        context = (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        )
        with context:
            (
                fused,
                w_outputs,
                ip_outputs,
                op_outputs,
                _fusion_statistics,
            ) = self.network(
                data, return_experts=True
            )
            loss = self.loss(fused, target)

        if self.enable_deep_supervision:
            fused_high = fused[0]
            w_high = w_outputs[0]
            ip_high = ip_outputs[0]
            op_high = op_outputs[0]
            target_high = target[0]
        else:
            fused_high = fused
            w_high = w_outputs
            ip_high = ip_outputs
            op_high = op_outputs
            target_high = target

        axes = [0] + list(range(2, fused_high.ndim))
        if self.label_manager.has_regions:
            prediction_onehot = (torch.sigmoid(fused_high) > 0.5).long()
        else:
            output_seg = fused_high.argmax(1)[:, None]
            prediction_onehot = torch.zeros(
                fused_high.shape,
                device=fused_high.device,
                dtype=torch.float16,
            )
            prediction_onehot.scatter_(1, output_seg, 1)

        diagnostic_target = target_high.clone()
        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                valid_mask = diagnostic_target != self.label_manager.ignore_label
                metric_target = target_high.clone()
                metric_target[metric_target == self.label_manager.ignore_label] = 0
                mask = valid_mask.float()
            else:
                valid_mask = None
                if target_high.dtype == torch.bool:
                    mask = ~target_high[:, -1:]
                else:
                    mask = 1 - target_high[:, -1:]
                metric_target = target_high[:, :-1]
        else:
            valid_mask = None
            mask = None
            metric_target = target_high

        tp, fp, fn, _ = get_tp_fp_fn_tn(
            prediction_onehot, metric_target, axes=axes, mask=mask
        )
        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        empty_counts = np.zeros(
            (len(self._CONFIDENCE_BIN_EDGES) - 1, 4), dtype=np.int64
        )
        ip_counts = empty_counts.copy()
        op_counts = empty_counts.copy()
        if not self.label_manager.has_regions:
            network = self._unwrapped_network()
            gains = network.last_confidence_gains
            if gains is None:
                raise RuntimeError("validation confidence gains were not recorded")
            ip_counts = self._confidence_repair_counts(
                w_high, ip_high, gains["ip"], diagnostic_target, valid_mask
            )
            op_counts = self._confidence_repair_counts(
                w_high, op_high, gains["op"], diagnostic_target, valid_mask
            )

        return {
            "loss": loss.detach().cpu().numpy(),
            "tp_hard": tp_hard,
            "fp_hard": fp_hard,
            "fn_hard": fn_hard,
            "ip_confidence_repair_counts": ip_counts,
            "op_confidence_repair_counts": op_counts,
        }

    @staticmethod
    def _sum_ddp_array(value: np.ndarray) -> np.ndarray:
        if not dist.is_available() or not dist.is_initialized():
            return value
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, value)
        return np.stack(gathered, axis=0).sum(axis=0)

    def _write_confidence_diagnostics(
        self, modality: str, counts: np.ndarray
    ) -> None:
        path = Path(self.output_folder) / "confidence_repair_diagnostics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.is_file()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not exists:
                writer.writerow(
                    [
                        "epoch", "modality", "bin_low", "bin_high",
                        "disagreement", "repair", "harm", "neither_correct",
                        "repair_probability", "harm_probability", "net_probability",
                    ]
                )
            for index, (low, high) in enumerate(
                zip(self._CONFIDENCE_BIN_EDGES[:-1], self._CONFIDENCE_BIN_EDGES[1:])
            ):
                disagreement, repair, harm, neither = [
                    int(value) for value in counts[index]
                ]
                denominator = max(disagreement, 1)
                writer.writerow(
                    [
                        self.current_epoch,
                        modality,
                        low,
                        high,
                        disagreement,
                        repair,
                        harm,
                        neither,
                        repair / denominator,
                        harm / denominator,
                        (repair - harm) / denominator,
                    ]
                )

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        outputs = collate_outputs(val_outputs)
        for modality in ("ip", "op"):
            key = f"{modality}_confidence_repair_counts"
            counts = np.asarray(outputs[key]).sum(axis=0)
            counts = self._sum_ddp_array(counts)
            if self.local_rank == 0:
                self._write_confidence_diagnostics(modality, counts)
                total = counts.sum(axis=0)
                disagreement = max(int(total[0]), 1)
                self.print_to_log_file(
                    f"Confidence diagnostic {modality.upper()}: "
                    f"disagreement={int(total[0])} "
                    f"repair={int(total[1]) / disagreement:.4f} "
                    f"harm={int(total[2]) / disagreement:.4f} "
                    f"net={(int(total[1]) - int(total[2])) / disagreement:.4f}"
                )

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        if self.local_rank == 0:
            self.print_to_log_file(
                "Expert losses "
                f"optimization={np.mean([x['optimization_loss'] for x in train_outputs]):.4f} "
                f"W={np.mean([x['w_loss'] for x in train_outputs]):.4f} "
                f"IP={np.mean([x['ip_loss'] for x in train_outputs]):.4f} "
                f"OP={np.mean([x['op_loss'] for x in train_outputs]):.4f}"
            )
            self.print_to_log_file(
                "Fusion protection "
                f"penalty={np.mean([x['w_anchor_penalty'] for x in train_outputs]):.6f} "
                f"weights(W/IP/OP)="
                f"{np.mean([x['mean_w_weight'] for x in train_outputs]):.4f}/"
                f"{np.mean([x['mean_ip_weight'] for x in train_outputs]):.4f}/"
                f"{np.mean([x['mean_op_weight'] for x in train_outputs]):.4f}"
            )

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.dixon_training_mode != "scratch":
            self.print_to_log_file(
                "Group learning rates:",
                [
                    round(float(group["lr"]), 8)
                    for group in self.optimizer.param_groups
                ],
            )

    @staticmethod
    def build_network_architecture(
        plans_manager,
        configuration_manager,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=True,
    ):
        if num_input_channels != 4:
            raise ValueError(
                "nnUNetTrainer_DixonEvidenceCalibratedMoE_4CH requires "
                "[W, synthetic_F, synthetic_IP, synthetic_OP]"
            )

        def build_one_expert():
            return get_network_from_plans(
                configuration_manager.network_arch_class_name,
                configuration_manager.network_arch_init_kwargs,
                configuration_manager.network_arch_init_kwargs_req_import,
                1,
                num_output_channels,
                allow_init=True,
                deep_supervision=enable_deep_supervision,
            )

        return DixonEvidenceCalibratedMoE4CH(
            unet_w=build_one_expert(),
            unet_ip=build_one_expert(),
            unet_op=build_one_expert(),
            num_classes=num_output_channels,
            modality_dropout={"ip": 0.08, "op": 0.15},
            confidence_thresholds=(0.20, 0.20),
            confidence_temperatures=(0.12, 0.12),
            initial_w_anchor_logit=2.0,
            minimum_w_weight=0.55,
            local_score_limit=2.0,
            case_score_limit=1.5,
            class_score_limit=0.75,
            record_fusion_maps=False,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        network = self.network.module if self.is_ddp else self.network
        if isinstance(network, OptimizedModule):
            network = network._orig_mod
        network.unet_w.decoder.deep_supervision = enabled
        network.unet_ip.decoder.deep_supervision = enabled
        network.unet_op.decoder.deep_supervision = enabled
