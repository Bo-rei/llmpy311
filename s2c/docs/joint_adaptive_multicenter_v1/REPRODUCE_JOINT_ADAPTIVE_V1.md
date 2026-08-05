# joint_adaptive_multicenter_v1 复现实验

## 环境

- 项目：`s2c`
- Conda：`bo`
- Python：3.11.13
- protocol：`protocol_v2_textoir_v1`
- 推荐命令前缀：`env -u PYTHONSTARTUP PYTHONPATH=src CONDA_NO_PLUGINS=true conda run -n bo`

## 运行

先确认 `repair6` provenance 已冻结，再使用 resume 汇总三个 seed：

```bash
cd s2c
export JOINT_ADAPTIVE_ATTEMPT=repair6
env -u PYTHONSTARTUP PYTHONPATH=src CONDA_NO_PLUGINS=true conda run -n bo \
  python scripts/experiments/run_joint_adaptive_multicenter_v1.py run \
  --config configs/experiments/protocol_v2_textoir_v1/joint_adaptive_multicenter_v1.yaml \
  --resume
```

生成汇总并校验：

```bash
env -u PYTHONSTARTUP PYTHONPATH=src CONDA_NO_PLUGINS=true conda run -n bo \
  python scripts/experiments/run_joint_adaptive_multicenter_v1.py summarize \
  --config configs/experiments/protocol_v2_textoir_v1/joint_adaptive_multicenter_v1.yaml

env -u PYTHONSTARTUP PYTHONPATH=src CONDA_NO_PLUGINS=true conda run -n bo \
  python scripts/experiments/run_joint_adaptive_multicenter_v1.py verify \
  --config configs/experiments/protocol_v2_textoir_v1/joint_adaptive_multicenter_v1.yaml

env -u PYTHONSTARTUP PYTHONPATH=src CONDA_NO_PLUGINS=true conda run -n bo \
  python scripts/experiments/analyze_joint_adaptive_v1.py --attempt repair6
```

## 结果位置

完整本地 artifact：

`../artifacts/s2c/runs/protocol_v2_textoir_v1/joint_adaptive_multicenter_v1/repair6/`

关键文件：

- `JOINT_ADAPTIVE_PROVENANCE.json`：代码、配置、模型和初始 checkpoint 哈希；
- `JOINT_ADAPTIVE_CODE.patch`：当前工作树相对 base commit 的代码快照；
- `runs/seed_*/split_events.json`：候选出生、训练、接受/拒绝原因；
- `runs/seed_*/metrics.json`：最终测试指标；
- `analysis/JOINT_ADAPTIVE_BASELINES.csv`：K=1、K=2 union 和 parent-guard 诊断；
- `analysis/JOINT_ADAPTIVE_PAIRED.csv`：同 seed 配对差值。

本阶段不把 StackOverflow 原文、embedding、checkpoint 或逐样本预测提交到 Git。
