#!/usr/bin/env bash
set -euo pipefail
fold="${1:?Usage: bash train.sh FOLD [--c]}"
[[ "$fold" =~ ^[0-4]$ ]] || { echo 'Fold must be 0..4' >&2; exit 2; }
resume=()
if [[ "${2:-}" == --c ]]; then
  resume=(--c)
elif [[ $# -gt 1 ]]; then
  echo 'Optional second argument must be --c' >&2; exit 2
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-8}"
cd "$(dirname "$0")"
python install.py
python create_plans.py
nnUNetv2_train 129 3d_fullres "$fold" -tr nnUNetTrainer_PHNetOriginal_1000ep -p nnUNetPlans_PHNetOriginal "${resume[@]}"
