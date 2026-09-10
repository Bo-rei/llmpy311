# 复现 consistency_gate_v1

```bash
cd s2c
env -u PYTHONSTARTUP PYTHONPATH=src CONDA_NO_PLUGINS=true conda run -n bo \
  python scripts/experiments/run_consistency_gate_v1.py run \
  --config configs/experiments/protocol_v2_textoir_v1/consistency_gate_v1.yaml \
  --resume

env -u PYTHONSTARTUP PYTHONPATH=src CONDA_NO_PLUGINS=true conda run -n bo \
  python scripts/experiments/run_consistency_gate_v1.py summarize \
  --config configs/experiments/protocol_v2_textoir_v1/consistency_gate_v1.yaml

env -u PYTHONSTARTUP PYTHONPATH=src CONDA_NO_PLUGINS=true conda run -n bo \
  python scripts/experiments/run_consistency_gate_v1.py verify \
  --config configs/experiments/protocol_v2_textoir_v1/consistency_gate_v1.yaml
```

所有 checkpoint 来自已完成的 RACAL Trainable K=1 阶段；本阶段不训练新 encoder，不增加中心。
阈值和冲突容忍度只由 Known calibration 选择，test OOS 不用于选择。
