"""Default official WNet3D architecture, adapted to nnU-Net v2 (1000 epochs).
Source: user-provided nnWNet-main.zip, default nnUNetTrainer_WNet3D.py.
Network topology, channel counts, normalization, pooling mixer and initialization
are preserved. This is NOT the M/S variant. No original files need overwriting.
Requires torch, timm, nnunetv2 and dynamic_network_architectures.
"""
import torch
import torch.nn as nn
try:
    from timm.layers import DropPath, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, trunc_normal_


BNNorm3d = nn.BatchNorm3d
LNNorm = nn.LayerNorm
Activation = nn.GELU

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            BNNorm3d(ch_out),
            Activation()
        )

    def forward(self, x):
        x = self.up(x)
        return x

class down_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(down_conv, self).__init__()
        self.down = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=False),
            BNNorm3d(ch_out),
            Activation()
        )

    def forward(self, x):
        x = self.down(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, groups=1):
        super(ResBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = nn.Conv3d(inplanes, planes, 3, stride=1, groups=groups, padding=1)
        self.bn1 = BNNorm3d(planes)
        self.act = Activation()
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=1, groups=groups, padding=1)
        self.bn2 = BNNorm3d(planes)

        if self.inplanes != self.planes:
            self.down = nn.Sequential(
                nn.Conv3d(inplanes, planes, 1, stride=1),
                BNNorm3d(planes)
            )
    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)

        if self.inplanes != self.planes:
            identity = self.down(x)

        out = self.bn2(out) + identity
        out = self.act(out)

        return out


class OPE(nn.Module):
    def __init__(self, inplanes, planes, groups=1):
        super(OPE, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, inplanes, 3, stride=1, groups=groups, padding=1)
        self.bn1 = BNNorm3d(inplanes)
        self.act = Activation()
        self.down = down_conv(inplanes, planes)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.down(out)

        return out


class local_block(nn.Module):
    def __init__(self, inplanes, hidden_planes, planes, groups=1, down_or_up=None):
        super(local_block, self).__init__()
        if down_or_up is None:
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
                # ResBlock(inplanes=planes, planes=planes, groups=groups),
            )

        elif down_or_up == 'down':
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
                # ResBlock(inplanes=planes, planes=planes, groups=groups),
                down_conv(hidden_planes, planes)
            )
        elif down_or_up == 'up':
            self.BasicBlock = nn.Sequential(
                ResBlock(inplanes=inplanes, planes=hidden_planes, groups=groups),
                # ResBlock(inplanes=planes, planes=planes, groups=groups),
                up_conv(hidden_planes, planes),
            )

    def forward(self, x):
        out = self.BasicBlock(x)
        return out

class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool_size = pool_size
        self.pool = nn.AvgPool3d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
        self.pool_ = nn.AvgPool3d(1, stride=1, padding=0, count_include_pad=False)
    def forward(self, x):
       if x.shape[2] < 3:
           return self.pool_(x) - x
       return self.pool(x) - x

class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()

        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = Activation()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class global_block(nn.Module):
    def __init__(self, in_dim, dim, num_heads, pool_size=3, mlp_ratio=4., drop=0., drop_path=0., sr_ratio=1):
        super().__init__()

        self.in_dim = in_dim
        self.dim = dim

        self.proj = nn.Conv3d(in_dim, dim, kernel_size=3,  padding=1)
        self.norm1 = GroupNorm(dim)
        self.attn = Pooling(pool_size=pool_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = GroupNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop)

    def forward(self, x):
        x = self.proj(x)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class WNet3D(nn.Module):
    def __init__(self, in_channel, num_classes, deep_supervised, layer_channel=[24, 48, 96, 192, 256], global_dim=[12, 24, 48, 96, 128], num_heads=[1, 2, 4, 8], sr_ratio=[8, 4, 2, 1], fusion_method='channel'):
        super(WNet3D, self).__init__()

        self.deep_supervised = deep_supervised

        self.input_l0 = nn.Sequential(
            nn.Conv3d(in_channel, layer_channel[0], kernel_size=3, stride=1, padding=1),
            BNNorm3d(layer_channel[0]),
            Activation(),
            nn.Conv3d(layer_channel[0], layer_channel[0], kernel_size=3, stride=1, padding=1),
            BNNorm3d(layer_channel[0]),
            Activation()
        )

        self.encoder1_l1_local = OPE(layer_channel[0], layer_channel[1])
        self.encoder1_l1_global = global_block(layer_channel[0], global_dim[0], num_heads=num_heads[0], sr_ratio=sr_ratio[0])

        self.encoder1_l2_local = OPE(layer_channel[1], layer_channel[2])
        self.encoder1_l2_global = global_block(layer_channel[1], global_dim[1], num_heads=num_heads[1], sr_ratio=sr_ratio[1])

        self.encoder1_l3_local = OPE(layer_channel[2], layer_channel[3])
        self.encoder1_l3_global = global_block(layer_channel[2], global_dim[2], num_heads=num_heads[2], sr_ratio=sr_ratio[2])

        self.encoder1_l4_local = OPE(layer_channel[3], layer_channel[4])
        self.encoder1_l4_global = global_block(layer_channel[3], global_dim[3], num_heads=num_heads[3], sr_ratio=sr_ratio[3])

        self.decoder1_l4_local = local_block(layer_channel[4], layer_channel[4], layer_channel[3], down_or_up='up')
        self.decoder1_l4_global = global_block(layer_channel[4], global_dim[4], num_heads=num_heads[3], sr_ratio=sr_ratio[3])

        self.decoder1_l3_local = local_block(layer_channel[3] + global_dim[3], layer_channel[3], layer_channel[2], down_or_up='up')
        self.decoder1_l3_global = global_block(layer_channel[3] + global_dim[3], global_dim[3], num_heads=num_heads[2], sr_ratio=sr_ratio[2])

        self.decoder1_l2_local = local_block(layer_channel[2] + global_dim[2], layer_channel[2], layer_channel[1], down_or_up='up')
        self.decoder1_l2_global = global_block(layer_channel[2] + global_dim[2], global_dim[2], num_heads=num_heads[1], sr_ratio=sr_ratio[1])

        self.decoder1_l1_local = local_block(layer_channel[1] + global_dim[1], layer_channel[1], layer_channel[0], down_or_up='up')
        self.decoder1_l1_global = global_block(layer_channel[1] + global_dim[1], global_dim[1], num_heads=num_heads[0], sr_ratio=sr_ratio[0])

        self.encoder2_l1_local = local_block(layer_channel[0] + global_dim[0], layer_channel[0], layer_channel[1], down_or_up='down')
        self.encoder2_l1_global = global_block(layer_channel[0] + global_dim[0], global_dim[0], num_heads=num_heads[0], sr_ratio=sr_ratio[0])

        self.encoder2_l2_local = local_block(layer_channel[1] + global_dim[1], layer_channel[1], layer_channel[2], down_or_up='down')
        self.encoder2_l2_global = global_block(layer_channel[1] + global_dim[1], global_dim[1], num_heads=num_heads[1], sr_ratio=sr_ratio[1])

        self.encoder2_l3_local = local_block(layer_channel[2] + global_dim[2], layer_channel[2], layer_channel[3], down_or_up='down')
        self.encoder2_l3_global = global_block(layer_channel[2] + global_dim[2], global_dim[2], num_heads=num_heads[2], sr_ratio=sr_ratio[2])

        self.encoder2_l4_local = local_block(layer_channel[3] + global_dim[3], layer_channel[3], layer_channel[4], down_or_up='down')
        self.encoder2_l4_global = global_block(layer_channel[3] + global_dim[3], global_dim[3], num_heads=num_heads[3], sr_ratio=sr_ratio[3])

        self.decoder2_l4_local_output = nn.Conv3d(layer_channel[4], num_classes, kernel_size=1, stride=1, padding=0)
        self.decoder2_l4_local = local_block(layer_channel[4] + global_dim[4], layer_channel[4], layer_channel[3], down_or_up='up')

        self.decoder2_l3_local_output = nn.Conv3d(layer_channel[3], num_classes, kernel_size=1, stride=1, padding=0)
        self.decoder2_l3_local = local_block(layer_channel[3] + global_dim[3], layer_channel[3], layer_channel[2], down_or_up='up')

        self.decoder2_l2_local_output = nn.Conv3d(layer_channel[2], num_classes, kernel_size=1, stride=1, padding=0)
        self.decoder2_l2_local = local_block(layer_channel[2] + global_dim[2], layer_channel[2], layer_channel[1], down_or_up='up')

        self.decoder2_l1_local_output = nn.Conv3d(layer_channel[1], num_classes, kernel_size=1, stride=1, padding=0)
        self.decoder2_l1_local = local_block(layer_channel[1] + global_dim[1], layer_channel[1], layer_channel[0], down_or_up='up')
        self.output_l0 = nn.Sequential(
            local_block(layer_channel[0] + global_dim[0], layer_channel[0], layer_channel[0], down_or_up=None),
            nn.Conv3d(layer_channel[0], num_classes, kernel_size=1, stride=1, padding=0)
        )
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        # initialization
        if isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):

        outputs = []

        # encoder-decoder 1
        x_e1_l0 = self.input_l0(x)

        x_e1_l1_local = self.encoder1_l1_local(x_e1_l0)
        x_e1_l0_global = self.encoder1_l1_global(x_e1_l0)

        x_e1_l2_local = self.encoder1_l2_local(x_e1_l1_local)
        x_e1_l1_global = self.encoder1_l2_global(x_e1_l1_local)

        x_e1_l3_local = self.encoder1_l3_local(x_e1_l2_local)
        x_e1_l2_global = self.encoder1_l3_global(x_e1_l2_local)

        x_e1_l4_local = self.encoder1_l4_local(x_e1_l3_local)
        x_e1_l3_global = self.encoder1_l4_global(x_e1_l3_local)

        x_d1_l3_local = self.decoder1_l4_local(x_e1_l4_local)
        x_d1_l4_global = self.decoder1_l4_global(x_e1_l4_local)

        x_d1_l3 = torch.cat((x_d1_l3_local, x_e1_l3_global), dim=1)
        x_d1_l2_local = self.decoder1_l3_local(x_d1_l3)
        x_d1_l3_global = self.decoder1_l3_global(x_d1_l3)

        x_d1_l2 = torch.cat((x_d1_l2_local, x_e1_l2_global), dim=1)
        x_d1_l1_local = self.decoder1_l2_local(x_d1_l2)
        x_d1_l2_global = self.decoder1_l2_global(x_d1_l2)

        x_d1_l1 = torch.cat((x_d1_l1_local, x_e1_l1_global), dim=1)
        x_d1_l0_local = self.decoder1_l1_local(x_d1_l1)
        x_d1_l1_global = self.decoder1_l1_global(x_d1_l1)

        # encoder-decoder 2
        x_e2_l0 = torch.cat((x_d1_l0_local, x_e1_l0_global), dim=1)
        x_e2_l1_local = self.encoder2_l1_local(x_e2_l0)
        x_e2_l0_global = self.encoder2_l1_global(x_e2_l0)

        x_e2_l1 = torch.cat((x_e2_l1_local, x_d1_l1_global), dim=1)
        x_e2_l2_local = self.encoder2_l2_local(x_e2_l1)
        x_e2_l1_global = self.encoder2_l2_global(x_e2_l1)

        x_e2_l2 = torch.cat((x_e2_l2_local, x_d1_l2_global), dim=1)
        x_e2_l3_local = self.encoder2_l3_local(x_e2_l2)
        x_e2_l2_global = self.encoder2_l3_global(x_e2_l2)

        x_e2_l3 = torch.cat((x_e2_l3_local, x_d1_l3_global), dim=1)
        x_e2_l4_local = self.encoder2_l4_local(x_e2_l3)
        x_e2_l3_global = self.encoder2_l4_global(x_e2_l3)

        outputs.append(self.decoder2_l4_local_output(x_e2_l4_local))
        x_e2_l4 = torch.cat((x_e2_l4_local, x_d1_l4_global), dim=1)
        x_d2_l3_local = self.decoder2_l4_local(x_e2_l4)

        outputs.append(self.decoder2_l3_local_output(x_d2_l3_local))
        x_d2_l3 = torch.cat((x_d2_l3_local, x_e2_l3_global), dim=1)
        x_d2_l2_local = self.decoder2_l3_local(x_d2_l3)

        outputs.append(self.decoder2_l2_local_output(x_d2_l2_local))
        x_d2_l2 = torch.cat((x_d2_l2_local, x_e2_l2_global), dim=1)
        x_d2_l1_local = self.decoder2_l2_local(x_d2_l2)

        outputs.append(self.decoder2_l1_local_output(x_d2_l1_local))
        x_d2_l1 = torch.cat((x_d2_l1_local, x_e2_l1_global), dim=1)
        x_d2_l0_local = self.decoder2_l1_local(x_d2_l1)

        x_d2_l0 = torch.cat((x_d2_l0_local, x_e2_l0_global), dim=1)
        outputs.append(self.output_l0(x_d2_l0))
        outputs = outputs[::-1]

        if self.deep_supervised:
            r = outputs
        else:
            r = outputs[0]
        return r


from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from dynamic_network_architectures.initialization.weight_init import InitWeights_He


class nnUNetTrainer_WNet3D_1000ep(nnUNetTrainer):
    """Single-GPU reference configuration; inherit base constructor unchanged.

    Keeping the installed nnU-Net constructor avoids version-specific
    unpack_dataset/device argument and checkpoint init_args incompatibilities.
    """

    def initialize(self):
        # Run before base initialization builds the optimizer and scheduler.
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.enable_deep_supervision = True
        self.save_every = 50

        patch = tuple(int(v) for v in self.configuration_manager.patch_size)
        if len(patch) != 3 or any(v < 16 or v % 16 for v in patch):
            raise ValueError(
                'WNet3D requires a 3D patch with each dimension >=16 and '
                f'divisible by 16; got {patch}. No automatic patch change.'
            )
        super().initialize()
        self.print_to_log_file(
            'WNet3D default: epochs=1000, train/val iterations=250/50, '
            'SGD lr=0.01 momentum=0.99 Nesterov, weight_decay=3e-5.'
        )
        self.print_to_log_file(
            'WNet3D deep supervision scales:', self._get_deep_supervision_scales()
        )

    @staticmethod
    def build_network_architecture(*args, **kwargs):
        """Accept the common old and new nnU-Net v2 builder interfaces.

        New: (architecture_class_name, arch_init_kwargs,
              arch_init_kwargs_req_import, num_input_channels,
              num_output_channels, enable_deep_supervision)
        Old: (plans_manager, configuration_manager, num_input_channels,
              num_output_channels, enable_deep_supervision=True)
        Architecture metadata is intentionally not used: this Trainer constructs
        the official fixed WNet3D topology, rather than the planned plain UNet.
        """
        if args:
            if isinstance(args[0], str):
                names = ('architecture_class_name', 'arch_init_kwargs',
                         'arch_init_kwargs_req_import', 'num_input_channels',
                         'num_output_channels', 'enable_deep_supervision')
            else:
                names = ('plans_manager', 'configuration_manager',
                         'num_input_channels', 'num_output_channels',
                         'enable_deep_supervision')
            if len(args) > len(names):
                raise TypeError('Unexpected nnU-Net network builder arguments')
            values = dict(zip(names, args))
            duplicate = values.keys() & kwargs.keys()
            if duplicate:
                raise TypeError(f'Duplicate builder arguments: {duplicate}')
            values.update(kwargs)
        else:
            values = kwargs
        if not {'num_input_channels', 'num_output_channels'} <= values.keys():
            raise TypeError('Network builder requires input and output channel counts')
        model = WNet3D(
            in_channel=int(values['num_input_channels']),
            num_classes=int(values['num_output_channels']),
            deep_supervised=bool(values.get('enable_deep_supervision', True)),
        )
        # Preserve the actual initialization used by the official Trainer.
        model.apply(InitWeights_He(1e-2))
        return model

    def _get_deep_supervision_scales(self):
        # WNet outputs five isotropic scales; do not use baseline plans strides.
        if not self.enable_deep_supervision:
            return None
        return [[1.0 / (2 ** i)] * 3 for i in range(5)]

    def set_deep_supervision_enabled(self, enabled: bool):
        # Unwrap both DDP and torch.compile without assuming wrapper order.
        model = self.network
        while True:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            elif hasattr(model, '_orig_mod'):
                model = model._orig_mod
            else:
                break
        if not isinstance(model, WNet3D):
            raise TypeError(f'Expected WNet3D, got {type(model).__name__}')
        model.deep_supervised = bool(enabled)

    def configure_optimizers(self):
        # Explicit schedule ensures the learning rate decays over all 1000 epochs.
        optimizer = torch.optim.SGD(
            self.network.parameters(), lr=self.initial_lr,
            weight_decay=self.weight_decay, momentum=0.99, nesterov=True,
        )
        scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, scheduler
