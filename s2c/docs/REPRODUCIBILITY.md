# 可复现性契约

## 数据与版本

唯一活动版本是 `protocol_v2_textoir_v1`。三数据集来自固定 TEXTOIR snapshot；
StackOverflow 仅允许本地训练/评估，不公开跟踪完整文本。`data/`、`../assets/` 和
`../artifacts/s2c/` 都是本地输入或原始产物，不能整体提交。

正式 run 必须同时保存 canonical manifest SHA、registry SHA、resolved config SHA、
encoder fingerprint、seed、K、distance、boundary、representation 和 `run_id`。没有
这些字段的结果不能进入论文表。

## 已冻结实验

E0/E1/E2/E3 以及历史 R1、MOGB、ADB/DA-ADB/DCLOOS 单元均有独立 artifact root；
`docs/EXPERIMENT_LEDGER.csv` 的 `repeat_policy=do_not_repeat` 行不得被新计划覆盖。
不要删除或重建 `../artifacts` 中的任何历史结果。

## 只读审计命令

```bash
cd /home/bo/bo01/llmpy311/s2c
python tools/analysis/audit_adaptive_k.py
python tools/analysis/diagnose_mogb_diff.py
python scripts/experiments/run_adaptive_split_merge.py --dry-run
python tools/analysis/audit_experiment_registry.py
python tools/maintenance/check_data_tracking.py
```

上述命令不训练模型；adaptive-K 和 MOGB 输出分别写入
`results/diagnostics/adaptive_k/`、`results/diagnostics/mogb_diff/`。

## 代码与环境

活动包只从 `src/protocol_v2/` 导入，禁止创建 `src/s2c/`。第三方 MOGB checkout
`third_party/mogb_official` 保持未修改；兼容层和来源说明必须分离。环境至少记录
Python、NumPy、SciPy、scikit-learn、PyTorch、Transformers 版本以及 GPU/CPU 设备。

## 结果隔离

轻量 CSV/JSON 放在 `results/`；模型、embedding、checkpoint、逐样本 predictions、
完整语料和日志只放本地 artifact。Gate-only 与完整 Cascade、官方 MOGB 与 MiniLM
fair adapter、真实 OOS 与 Known-only 结果必须分栏，不能仅因文件名相似而合并。
