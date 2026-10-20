# Dixon 局部优势探索与最终训练：严格执行手册

这套流程只使用 `checkpoint_final.pth`。任何一步失败都先停止，不要跳过检查直接训练。

## 0. 固定路径并安装代码

以下只需把 `BASE_MODEL` 与 `BASELINE_MODEL` 改成服务器上真实存在的目录；其余路径来自当前 Dataset135 环境。

```bash
cd /root/nnUNet

export RAW4=/root/autodl-tmp/nnUNet/nnUNet_raw/Dataset135_4CH
export PRE4=/root/autodl-tmp/nnUNet/nnUNet_preprocessed/Dataset135_4CH
export BASE_MODEL=/root/autodl-tmp/nnUNet/nnUNet_results/Dataset135_4CH/nnUNetTrainer_DixonIndependentAnchoredMoE_4CH__nnUNetPlans__3d_fullres
export BASELINE_MODEL=/root/autodl-tmp/nnUNet/Dataset129_1CH_1/nnUNetTrainer__nnUNetPlans__3d_fullres
export WORK=/root/autodl-tmp/test_evaluation_results/dixon_local_advantage
export PIPE=/root/nnUNet

mkdir -p "$WORK"

python - <<'PY'
import importlib
m = importlib.import_module("nnunetv2.training.nnUNetTrainer.nnUNetTrainer_DixonLocalAdvantageRouter_4CH")
print("Loaded module:", m.__file__)
print("Probe:", m.nnUNetTrainer_DixonLocalAdvantageRouterProbe_4CH)
print("Curriculum:", m.nnUNetTrainer_DixonLocalAdvantageRouterCurriculum_4CH)
PY

test -f "$BASE_MODEL/fold_0/checkpoint_final.pth"
test -f "$BASE_MODEL/fold_0/validation/summary.json"
test -f "$BASELINE_MODEL/fold_0/checkpoint_final.pth"
test -f "$BASELINE_MODEL/fold_0/validation/summary.json"
```

不要用 `类名.__file__`；`__file__` 属于 Python 模块。

## 1. 锁定病例、权重与几何

```bash
python "$PIPE/01_verify_experiment.py" \
  --images-dir "$RAW4/imagesTr" \
  --reference-dir "$PRE4/gt_segmentations" \
  --dataset-json "$RAW4/dataset.json" \
  --baseline-summary "$BASELINE_MODEL/fold_0/validation/summary.json" \
  --independent-summary "$BASE_MODEL/fold_0/validation/summary.json" \
  --baseline-checkpoint "$BASELINE_MODEL/fold_0/checkpoint_final.pth" \
  --independent-checkpoint "$BASE_MODEL/fold_0/checkpoint_final.pth" \
  --output "$WORK/verification_manifest.json"
```

继续条件：打印 `PASS`。脚本检查两模型病例集合相同、W/F/IP/OP 与标签 shape/affine 一致，并记录权重 SHA-256。

## 2. 从同一个 Independent final 权重导出四种完整体积预测

专家模式直接执行指定单专家，不会白算另外两个专家和 Router。

```bash
CUDA_VISIBLE_DEVICES=0 python "$PIPE/02_predict_independent_experts.py" \
  --model-dir "$BASE_MODEL" \
  --images-dir "$RAW4/imagesTr" \
  --manifest "$WORK/verification_manifest.json" \
  --output-dir "$WORK/expert_predictions" \
  --fold 0 \
  --checkpoint-name checkpoint_final.pth \
  --device cuda \
  --modes all w ip op \
  --num-processes 2 \
  --overwrite
```

正式统计保留 nnU-Net 默认镜像。显存不足时不要改 patch size；本命令默认已经把结果累加数组放在 CPU。

## 3. 计算专家互补性、缺失类别 FP 和 Oracle 上限

```bash
python "$PIPE/03_analyze_expert_oracle.py" \
  --predictions-dir "$WORK/expert_predictions" \
  --reference-dir "$PRE4/gt_segmentations" \
  --dataset-json "$RAW4/dataset.json" \
  --manifest "$WORK/verification_manifest.json" \
  --output-dir "$WORK/oracle" \
  --target-dice 0.890926

python -m json.tool "$WORK/oracle/oracle_summary.json"
```

决策规则：

- `STOP_ROUTER_REDESIGN`：专家没有足够互补空间，停止本路线。
- `GO_PROBE_ONLY`：只进入短 Probe，不能直接跑 1000 轮。
- `GO_FULL`：仍先跑 Probe；Probe 通过后才允许正式重训。

Oracle 按“病例-类别”看过真值后挑专家，只是上限，不是模型分数。

## 4. 定位 IP/OP 在哪里帮助或伤害 W

```bash
python "$PIPE/04_analyze_local_advantage.py" \
  --predictions-dir "$WORK/expert_predictions" \
  --reference-dir "$PRE4/gt_segmentations" \
  --dataset-json "$RAW4/dataset.json" \
  --manifest "$WORK/verification_manifest.json" \
  --output-dir "$WORK/local_advantage" \
  --boundary-width 2 \
  --volume-shell-width 12

column -s, -t < "$WORK/local_advantage/local_advantage_summary.csv"
```

继续条件：IP 或 OP 至少在 `interior`、`inner_boundary`、`outer_boundary` 中一个区域呈稳定正收益，且收益不是仅来自 `volume_shell`。`volume_shell` 是 FOV 边缘，不冒充滑窗 patch 边缘。

## 5. 75 轮 Router Probe（冻结三个成熟专家）

Probe 只回答“新 Router 能否把现成专家互补性转成完整体积分数”。新增参数以恒等映射初始化，所以 epoch 0 继承原 Independent 融合。

```bash
mkdir -p /root/autodl-tmp/nnUNet/nnUNet_results_router_probe

DIXON_TRAINING_MODE=scratch \
DIXON_SOURCE_CHECKPOINT="$BASE_MODEL/fold_0/checkpoint_final.pth" \
DIXON_PROBE_EPOCHS=75 \
DIXON_FUSION_LR=0.001 \
DIXON_ORACLE_MARGIN=0.05 \
nnUNet_results=/root/autodl-tmp/nnUNet/nnUNet_results_router_probe \
nnUNetv2_train 135 3d_fullres 0 -tr nnUNetTrainer_DixonLocalAdvantageRouterProbe_4CH
```

```bash
export PROBE_MODEL=/root/autodl-tmp/nnUNet/nnUNet_results_router_probe/Dataset135_4CH/nnUNetTrainer_DixonLocalAdvantageRouterProbe_4CH__nnUNetPlans__3d_fullres

python - "$PROBE_MODEL/fold_0/validation/summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("Probe checkpoint_final foreground mean Dice =", d["foreground_mean"]["Dice"])
PY
```

严格阈值：

- `< 0.8892`：停止，Router 方向失败。
- `0.8892–0.8902`：增益太弱，不投入正式 1000 轮。
- `>= 0.8902`：允许执行第 6 步。
- `>= 0.890926`：Probe 达到相对 baseline 的绝对 +0.01 目标，但仍需正式训练与五折验证。

不要给 Probe 使用 `-pretrained_weights`。原生 loader 要求所有键完全相同；这里用兼容的 `DIXON_SOURCE_CHECKPOINT`。

## 6. 一次 1000 轮课程式正式训练

这不是三次训练，而是同一个 optimizer、同一个 checkpoint 链和总计 1000 个 epoch：

| 阶段 | epoch | W 专家 | IP/OP 专家 | Router/fusion |
|---|---:|---|---|---|
| A 专家形成 | 0–199 | 直接监督、持续训练 | 直接监督、持续训练 | LR=0，不学习 |
| B Router 学习 | 200–499 | 继续直接监督；隔离 fused 梯度 | 继续直接监督；隔离 fused 梯度 | 学习局部/类别路由 |
| C 保守协同 | 500–999 | 继续直接监督；始终隔离 fused 梯度 | 继续直接监督；fused 梯度从 0 升至 0.1 | 持续学习，Oracle 权重逐渐减小 |

阶段 B/C 没有停止三个专家；W 完整接受 1000 轮直接监督。

```bash
mkdir -p /root/autodl-tmp/nnUNet/nnUNet_results_local_advantage

DIXON_TRAINING_MODE=scratch \
DIXON_NUM_EPOCHS=1000 \
DIXON_STAGE_A_END=200 \
DIXON_STAGE_B_END=500 \
DIXON_FUSION_LR=0.001 \
DIXON_ORACLE_MARGIN=0.05 \
nnUNet_results=/root/autodl-tmp/nnUNet/nnUNet_results_local_advantage \
nnUNetv2_train 135 3d_fullres 0 -tr nnUNetTrainer_DixonLocalAdvantageRouterCurriculum_4CH
```

中断后只能用完全相同的环境变量，加 `--c` 恢复。

## 7. fold 0 最终判定与五折

```bash
export FINAL_MODEL=/root/autodl-tmp/nnUNet/nnUNet_results_local_advantage/Dataset135_4CH/nnUNetTrainer_DixonLocalAdvantageRouterCurriculum_4CH__nnUNetPlans__3d_fullres

python - "$FINAL_MODEL/fold_0/validation/summary.json" "$BASELINE_MODEL/fold_0/validation/summary.json" <<'PY'
import json, sys
new = json.load(open(sys.argv[1]))["foreground_mean"]["Dice"]
base = json.load(open(sys.argv[2]))["foreground_mean"]["Dice"]
print(f"baseline={base:.6f}")
print(f"new={new:.6f}")
print(f"absolute_gain={new-base:+.6f} ({100*(new-base):+.3f} percentage points)")
print("PASS_1_PERCENT_POINT=", new - base >= 0.01)
PY
```

只有 fold 0 达到预设阈值后才跑其余折：

```bash
for FOLD in 1 2 3 4; do
  DIXON_TRAINING_MODE=scratch \
  DIXON_NUM_EPOCHS=1000 \
  DIXON_STAGE_A_END=200 \
  DIXON_STAGE_B_END=500 \
  DIXON_FUSION_LR=0.001 \
  DIXON_ORACLE_MARGIN=0.05 \
  nnUNet_results=/root/autodl-tmp/nnUNet/nnUNet_results_local_advantage \
  nnUNetv2_train 135 3d_fullres "$FOLD" -tr nnUNetTrainer_DixonLocalAdvantageRouterCurriculum_4CH
done
```

最终论文比较报告五折 mean±SD，并对相同病例做 paired bootstrap 或 Wilcoxon。fold 0 探索阈值不能替代五折结论，也不能保证必然获得 +1 个百分点。
