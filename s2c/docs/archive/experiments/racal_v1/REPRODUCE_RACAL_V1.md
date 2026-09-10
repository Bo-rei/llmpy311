# RACAL-v1 第一阶段复现

```bash
cd /home/bo/bo01/llmpy311/s2c

python scripts/experiments/racal_v1/run_racal_v1.py \
  --config configs/experiments/protocol_v2_textoir_v1/racal_v1.yaml \
  --dry-run

python scripts/experiments/racal_v1/run_racal_v1.py \
  --config configs/experiments/protocol_v2_textoir_v1/racal_v1.yaml \
  --method frozen_k1 --seeds 13 42 87 --resume

python scripts/experiments/racal_v1/run_racal_v1.py \
  --config configs/experiments/protocol_v2_textoir_v1/racal_v1.yaml \
  --method trainable_k1 --seed 42

python scripts/experiments/racal_v1/run_racal_v1.py \
  --config configs/experiments/protocol_v2_textoir_v1/racal_v1.yaml \
  --method trainable_k1 --seeds 13 87 --resume

python scripts/experiments/racal_v1/verify_racal_v1.py \
  --run-root ../artifacts/s2c/runs/protocol_v2_textoir_v1/racal_v1

python scripts/experiments/racal_v1/summarize_racal_v1.py \
  --run-root ../artifacts/s2c/runs/protocol_v2_textoir_v1/racal_v1 \
  --output-dir results/diagnostics/racal_v1
```

所有输出必须包含 `test_used_for_selection=false`、输入 sample-id hash、E2 manifest hash、
模型文件 hash 和 checkpoint hash。完整 checkpoint、文本和逐样本预测只保留在本地 artifact。
