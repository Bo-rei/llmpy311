# s2c 项目契约

这份文件只说明当前项目怎么组织、哪些结果可信，以及阅读时不能越过的边界。历史设计和旧实验不再作为当前入口。

## 系统边界

```text
文本
 └─ MiniLM Gate：Known / OOS
     └─ SmolLM Router：domain
         └─ SmolLM Expert：intent
```

研究主线是 `MiniLM 表示 → Known 局部支持结构 → OOS 可分性`。Known intent 分类只用于检查 false reject、多簇收益和完整系统传递，不是本轮独立方法贡献。

## 工作区布局

```text
../assets/       数据集、split、预训练模型
../artifacts/    被 Git 忽略的模型和实验产物
s2c/src/         可复用 pipeline、Gate、Router、Expert 实现
s2c/tools/       训练、评价、分析入口
s2c/tests/       协议和回归测试
s2c/configs/     运行配置和实验登记
s2c/docs/        仅四份活动文档；历史材料在 docs/archive/
```

路径由 `src.runtime.WorkspacePaths` 推导。不要在仓库内复制 `data/`、`models/` 或 `outputs/`。

## 当前可运行入口

| 工作 | 入口 | 当前状态 |
| --- | --- | --- |
| 数据准备/基础流程 | `python -m src.cli` | 可运行 |
| Gate-only 机制实验 | `python -m tools.experiments.cluster_separability` | 结果冻结 |
| 下游组件预检 | `python tools/train/run_cascade_components.py` | 9/9 ready |
| Gate 适配预检 | `python tools/eval/prepare_cascade_gates.py` | 9/9 ready |
| 完整 Cascade 预检 | `python tools/eval/run_cascade_matrix.py` | 36/36 complete |
| 汇总与配对分析 | `python tools/analysis/export_cascade_repair_summary.py` | 当前 main 可运行 |
| 结果登记审计 | `python tools/analysis/audit_experiment_registry.py` | 当前 main 可运行 |

活动入口和历史兼容脚本的分类见 `configs/active_entrypoints.json`。未引用脚本只生成报告，不自动删除。

## Banking77/StackOverflow 是否已经修复

已经修复并重新运行，不是待办事项：

1. 单域数据集不再训练无意义的 1-class SmolLM Router，而是使用 manifest 声明的 constant router。
2. Expert 只按 `MANIFEST.json` 中声明的 domain 训练，避免把残留目录误当成真实 domain。
3. Banking77/StackOverflow seed13、87 使用本轮 GPU 训练组件，seed42 使用已审计组件；四种 Gate 共用同一 dataset/seed 的下游组件。
4. Expert 单点 test accuracy 约为 Banking77 `0.849`、StackOverflow `0.874`；旧 smoke 中的异常低数字不属于当前主结果。

因此当前若看到 Known macro-F1 的变化，应先按 `cascade_banking_tradeoff.csv` 和 `cascade_error_decomposition_summary.csv` 判断是 Gate 的 false reject/false accept 权衡，而不是再次误判为下游组件坏掉。

## 证据优先级

```text
单元 eval_results.json / run_manifest.json
  > matrix_manifest.json / 汇总 CSV
  > 当前源码和测试
  > 活动文档
  > docs/archive/ 中的历史说明和旧论文稿
```

论文数字必须能追溯到 dataset、KIR、seed、split、Gate 配置、checkpoint、validation 选择规则和源码 commit。

## 当前研究边界

- 主数据集：CLINC150、BANKING77-OOS、StackOverflow。
- 当前完整系统矩阵只冻结 KIR50、seed `{13,42,87}`、四种代表 Gate，共 36 个单元；不扩展 KIR25/75。
- 不新增 Gate、K、encoder 或普通 baseline 家族。
- MOGB 只有协议审计时，不把审计 JSON 当成性能结果。
- 当前 `bo` 环境实际 sklearn 版本为 `1.8.0`，baseline 反序列化审计无 warning；不因不存在的 warning 重训。
- 完整 Pipeline 的系统级结论必须同时报告 OOS F1、Known macro-F1、accuracy、ID Recall 和 Gate/Router/Expert 错误分解。

## 结果位置

```text
../artifacts/s2c/outputs/experiments/cascade_full/gpu_kir50/
├── component_plan.json
├── gates/gate_manifest.json
├── evaluations/matrix_manifest.json
├── cascade_summary.csv
├── cascade_gate_by_seed.csv
├── cascade_gate_summary.csv
├── cascade_error_decomposition.csv
├── cascade_error_decomposition_summary.csv
└── cascade_banking_tradeoff.csv
```

Gate-only、MiniLM、外部 hard-negative 和历史研究结果统一登记在 `configs/experiment_registry.yaml`；执行登记审计不会改变任何结果。
