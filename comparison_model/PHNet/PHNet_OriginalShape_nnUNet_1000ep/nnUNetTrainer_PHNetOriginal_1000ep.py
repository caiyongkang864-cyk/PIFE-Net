"""Native PHNet Synapse architecture in nnU-Net v2 with torch.compile.

Official network implementation is copied unchanged to _phnet_original_impl.
Only input/output channels adapt to Dataset129_1CH; the upstream nnU-Net v1
wrapper's axis permutation is implemented without requiring nnU-Net v1.
"""

import torch
from torch import nn
from torch._dynamo import OptimizedModule
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from monai.losses import DiceLoss

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer._phnet_original_impl.phnet import PHNet


class PHNetMRIOriginal(PHNet):
    def __init__(self, in_channels=1, out_channels=63):
        super().__init__(
            res_ratio=4.0,
            layers=(15, 4),
            in_channels=in_channels,
            out_channels=out_channels,
            embed_dims=(48, 96, 192, 192, 384),
            segment_dim=(12, 6),
            mlp_ratio=4.0,
            dropout_rate=0.3,
        )

    def forward(self, x):
        # Same axis permutation as upstream network/phnet_synapse.py:
        # nnU-Net [B,C,D,H,W] -> PHNet [B,C,H,W,D] -> nnU-Net logits.
        logits = super().forward(x.permute(0, 1, 3, 4, 2))
        return logits.permute(0, 1, 4, 2, 3)


class PHNetDiceCE(nn.Module):
    """Official run_net.py's MONAI DiceLoss + nn.CrossEntropyLoss."""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.dice(logits, labels) + self.cross_entropy(
            logits, labels.squeeze(1).long()
        )


class nnUNetTrainer_PHNetOriginal_1000ep(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(
        plans_manager, configuration_manager, num_input_channels,
        num_output_channels, enable_deep_supervision=True,
    ):
        if (num_input_channels, num_output_channels) != (1, 63):
            raise ValueError(f'Expected W-only 1 -> 63, got {num_input_channels} -> {num_output_channels}')
        return PHNetMRIOriginal(num_input_channels, num_output_channels)

    def _do_i_compile(self):
        return self.device.type == 'cuda'

    def _build_loss(self):
        return PHNetDiceCE()

    def configure_optimizers(self):
        optimizer = Adam(self.network.parameters(), lr=self.initial_lr)
        # The original MONAI training uses Adam at a fixed lr. nnU-Net calls
        # .step(epoch), so a constant LambdaLR preserves that recipe.
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        return optimizer, scheduler

    def initialize(self):
        if self.was_initialized:
            raise RuntimeError('Trainer already initialized')
        if self.is_ddp:
            raise ValueError('Use a single GPU for this package')
        if self.configuration_name != '3d_fullres':
            raise ValueError('Use 3d_fullres in nnUNetPlans_PHNetOriginal')
        if self.plans_manager.plans_name != 'nnUNetPlans_PHNetOriginal':
            raise ValueError('Use -p nnUNetPlans_PHNetOriginal')
        if tuple(self.configuration_manager.patch_size) != (48, 192, 192):
            raise ValueError('Expected nnU-Net patch (48,192,192), corresponding to native (192,192,48)')
        if self.configuration_manager.batch_size != 2:
            raise ValueError('Expected batch_size=2')
        if self.label_manager.has_regions or self.label_manager.ignore_label is not None:
            raise ValueError('Requires exclusive labels 0..62, no ignore label')
        if self.label_manager.num_segmentation_heads != 63:
            raise ValueError('Requires 63 output classes including background')

        self.enable_deep_supervision = False
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.initial_lr = 8e-5  # original run_net.py default
        self.save_every = 50
        super().initialize()

        if not isinstance(self.network, OptimizedModule):
            raise RuntimeError('torch.compile was requested but network is not compiled')
        if not isinstance(self.loss, PHNetDiceCE) or not isinstance(self.optimizer, Adam):
            raise RuntimeError('Original PHNet loss/optimizer was not installed')
        self.print_to_log_file(
            'PHNet original Synapse network: W-only, 63 logits; '
            'res_ratio=4, widths=(48,96,192,192,384), layers=(15,4), '
            'segment_dim=(12,6), dropout=0.3; '
            'native patch=(192,192,48), nnU-Net patch=(48,192,192), '
            'batch=2, epochs=1000, train=250/epoch, val=50/epoch; '
            'Adam lr=8e-5 constant, MONAI Dice+CE, no deep supervision, '
            'AMP per nnU-Net CUDA trainer; torch.compile ON; '
            f'parameters={sum(p.numel() for p in self.network.parameters()):,}'
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        self.enable_deep_supervision = False
