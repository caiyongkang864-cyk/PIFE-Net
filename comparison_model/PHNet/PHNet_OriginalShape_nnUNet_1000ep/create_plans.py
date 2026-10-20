"""Clone baseline plans with PHNet's official Synapse patch, sharing preprocessed data."""
import json
import os
from pathlib import Path


def main():
    root = Path(os.environ['nnUNet_preprocessed']) / 'Dataset129_1CH'
    source = root / 'nnUNetPlans.json'
    destination = root / 'nnUNetPlans_PHNetOriginal.json'
    plans = json.loads(source.read_text())
    cfg = plans['configurations']['3d_fullres']
    if tuple(cfg['patch_size']) != (48, 160, 192) or cfg['batch_size'] != 2:
        raise ValueError('Baseline plans changed: expected patch=(48,160,192), batch=2')
    identifier = cfg['data_identifier']
    if not (root / identifier).is_dir():
        raise FileNotFoundError(f'Preprocessed cases missing: {root / identifier}')
    plans['plans_name'] = 'nnUNetPlans_PHNetOriginal'
    cfg['patch_size'] = [48, 192, 192]
    cfg['batch_size'] = 2
    cfg['architecture'] = {
        'network_class_name': 'nnunetv2.training.nnUNetTrainer.nnUNetTrainer_PHNetOriginal_1000ep.PHNetMRIOriginal',
        'arch_kwargs': {'in_channels': 1, 'out_channels': 63},
        '_kw_requires_import': [],
    }
    if destination.exists():
        if json.loads(destination.read_text()) == plans:
            print('Existing matching plans:', destination)
            return
        raise RuntimeError(f'Existing plans differ: {destination}; inspect before replacing')
    destination.write_text(json.dumps(plans, indent=2) + '\n')
    print('Created:', destination)
    print('Shared preprocessed data:', identifier)


if __name__ == '__main__':
    main()
