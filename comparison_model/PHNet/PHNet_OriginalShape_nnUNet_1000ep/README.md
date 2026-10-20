# PHNet original Synapse architecture on MRISegmenter (nnU-Net v2)

This is a **new, separate experiment** from `nnUNetTrainer_PHNet_1000ep`.
The three official files `phnet.py`, `MLPP.py`, `basic_block.py` in
`phnet_original_impl/` are byte-for-byte copies of the uploaded PHNet repo.
No layers, feature widths, dropout, or segment sizes are changed. A wrapper
performs exactly the axis swap in upstream `phnet_synapse.py` without requiring
nnU-Net v1, and adapts only 1 input W channel and 63 output labels.

| Setting | Value |
| --- | --- |
| PHNet original Synapse patch (H,W,D) | `(192,192,48)` |
| nnU-Net v2 patch (D,H,W) | `(48,192,192)` |
| PHNet `res_ratio`, `layers`, `segment_dim` | `4`, `(15,4)`, `(12,6)` |
| PHNet widths, dropout | `(48,96,192,192,384)`, `0.3` |
| Budget | 1000 epochs, 250 train + 50 validation iterations/epoch |
| Batch | 2 |
| Optimizer/loss | Original Adam `8e-5` constant + MONAI Dice/CE |
| Precision/compile | nnU-Net CUDA AMP, `torch.compile` forced on |

The trainer uses the same nnU-Net v2 preprocessed **Dataset129_1CH** and
`splits_final.json`, random patch loader and augmentation as the baseline.
This is not the upstream MONAI training loop or an architecture-only ablation
versus the nnU-Net baseline. Patch size differs from the baseline's
`(48,160,192)` to preserve PHNet's original `(12,6)` segment design, so each
step processes 20% more voxels. Report that difference in any comparison.

## Setup

In the Python environment that imports your nnU-Net v2 and has PyTorch 2.8:

```bash
cd /root/PHNet_OriginalShape_nnUNet_1000ep
python -m pip install monai
export nnUNet_preprocessed=/root/autodl-tmp/nnUNet/nnUNet_preprocessed
export nnUNet_raw=/root/autodl-tmp/nnUNet/nnUNet_raw
export nnUNet_results=/root/autodl-tmp/nnUNet/nnUNet_results
python install.py
python create_plans.py
CUDA_VISIBLE_DEVICES=0 python benchmark.py
```

`create_plans.py` writes a separate
`Dataset129_1CH/nnUNetPlans_PHNetOriginal.json`; it retains the baseline
spacing and `nnUNetPlans_3d_fullres` data identifier, so it does not modify
baseline plans or re-preprocess the dataset. It requires the original baseline
plans entry to have patch `(48,160,192)` and batch 2. The plan is idempotent
and refuses to overwrite a different existing PHNet plan.

The benchmark runs *compiled* forward and backward at batch 2 and the full
original PHNet patch. It must pass on the intended GPU before training.
`torch.compile` may increase initial compile time and peak memory, and is not
guaranteed to speed this particular network. The project author did not
specify PyTorch 2.8 compilation in its original implementation.

## Train and resume

```bash
CUDA_VISIBLE_DEVICES=0 bash train.sh 0
CUDA_VISIBLE_DEVICES=0 bash train.sh 0 --c
```

Use folds 1, 2, 3, 4 in the same manner. `--c` resumes **only** checkpoints
from this trainer and this plans name. Do not mix folds/checkpoints from the
previous `(4,2)` network. Predictions must use both `-tr
nnUNetTrainer_PHNetOriginal_1000ep` and `-p nnUNetPlans_PHNetOriginal`.

This code has passed static syntax and source-equivalence checks. The current
workspace does not have PyTorch/GPU; the compiled benchmark and training must
be checked in your CUDA environment. Test performance only with held-out
full-volume predictions and the same evaluator as other methods.
