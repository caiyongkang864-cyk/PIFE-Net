"""Install this adaptation next to the imported nnU-Net v2 trainer."""
import importlib
import inspect
import shutil
from pathlib import Path

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def main():
    source = Path(__file__).resolve().parent
    destination = Path(inspect.getfile(nnUNetTrainer)).resolve().parent
    signature = inspect.signature(nnUNetTrainer.build_network_architecture)
    expected = ('plans_manager', 'configuration_manager', 'num_input_channels', 'num_output_channels')
    if tuple(signature.parameters)[:4] != expected:
        raise RuntimeError(f'Incompatible nnU-Net API: {signature}')

    trainer_name = 'nnUNetTrainer_PHNetOriginal_1000ep.py'
    shutil.copy2(source / trainer_name, destination / trainer_name)
    shutil.copytree(source / 'phnet_original_impl', destination / '_phnet_original_impl', dirs_exist_ok=True)
    importlib.invalidate_caches()
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_PHNetOriginal_1000ep import nnUNetTrainer_PHNetOriginal_1000ep
    print('Installed trainer:', inspect.getfile(nnUNetTrainer_PHNetOriginal_1000ep))
    print('Installed original network:', destination / '_phnet_original_impl')


if __name__ == '__main__':
    main()
