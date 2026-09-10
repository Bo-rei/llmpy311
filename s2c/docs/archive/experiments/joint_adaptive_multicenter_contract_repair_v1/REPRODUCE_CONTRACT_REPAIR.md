# 复现实验：joint_adaptive_multicenter_contract_repair_v1

## 环境

项目根：`s2c/`。为避免 `PYTHONSTARTUP` 导致的 torch double-free，使用：

```bash
env -u PYTHONSTARTUP PYTHONPATH=src CONDA_NO_PLUGINS=true conda run -n bo <command>
```

## 运行

```bash
export JOINT_ADAPTIVE_CONTRACT_ATTEMPT=repair3
python scripts/experiments/run_joint_adaptive_contract_repair_v1.py run \
  --config configs/experiments/protocol_v2_textoir_v1/joint_adaptive_multicenter_contract_repair_v1.yaml \
  --resume
python scripts/experiments/run_joint_adaptive_contract_repair_v1.py summarize \
  --config configs/experiments/protocol_v2_textoir_v1/joint_adaptive_multicenter_contract_repair_v1.yaml
python scripts/experiments/run_joint_adaptive_contract_repair_v1.py verify \
  --config configs/experiments/protocol_v2_textoir_v1/joint_adaptive_multicenter_contract_repair_v1.yaml
```

## 复现约束

- 只使用 StackOverflow、KIR=0.50、seed 13/42/87；
- 训练、候选结构选择和边界约束只读 Known train/calibration；
- K=1 父边界冻结；候选最多一个 split；
- 不使用 test OOS 选 epoch、中心、半径、阈值或 margin；
- 不修改 `joint_adaptive_multicenter_v1/repair6` 和 E2/E3/R1/RACAL artifacts；
- `repair3/JOINT_ADAPTIVE_CONTRACT_REPAIR_PROVENANCE.json` 是当前有效 provenance。
