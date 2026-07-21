# s2c 当前项目契约

这份文件是项目的“边界说明”，用于回答：代码在哪里、数据在哪里、结果是否可复现、哪些数字可以写进论文。

## 一句话

```text
文本 → Gate（Known/OOS）→ Router（domain）→ Expert（intent）
```

当前研究重点是冻结 MiniLM 语义空间中的 Known 多簇结构与 Known/OOS 可分性；Gate-only 机制实验是主体，完整 Cascade 是系统级验证。

## 工作区布局

从 `s2c/` 项目根目录看：

```text
../assets/       数据集、预训练模型和大文件
../artifacts/    被 .gitignore 忽略的实验产物
s2c/src/         可复用实现
s2c/tools/       训练、评价和实验编排入口
s2c/tests/       协议和回归测试
s2c/docs/        当前契约、实验说明和历史参考
```

不要在仓库内复制 `data/`、`models/` 或 `outputs/`；路径由 `src.runtime.WorkspacePaths` 统一解析。

## 当前可依赖的源码

| 任务 | 入口 | 状态 |
| --- | --- | --- |
| 数据准备/主流程 | `python -m src.cli` | main 可运行 |
| v19/v20/v21 Gate-only 研究 | `python -m tools.experiments.cluster_separability` | main 可运行 |
| Gate→Router→Expert 评价 | `tools/eval/eval_system_pipeline_v19.py` | main 可运行 |
| 本轮下游修复后的代表 Cascade | `tools/eval/run_cascade_repair.py` | main 可运行，输出到独立目录 |
| 完整 KIR50 Cascade 组件 | `tools/train/run_cascade_components.py` | 9 个 dataset×seed 组件均已 ready；以 `cascade_full/gpu_kir50/component_plan.json` 为准 |
| 完整 KIR50 Cascade Gate 适配 | `tools/eval/prepare_cascade_gates.py` | 9 个 Gate adapter 均已 ready；以 `cascade_full/gpu_kir50/gates/gate_manifest.json` 为准 |
| 完整 KIR50 Cascade 评价 | `tools/eval/run_cascade_matrix.py` | 36/36 完成；以 `cascade_full/gpu_kir50/evaluations/matrix_manifest.json` 为准 |
| CE-Recon/Cascade 收口入口 | 当前 `main` 工作树中的 `tools/eval/`、`tools/train/` 和 `tools/analysis/` | 当前工作树可运行；对应新增文件尚未形成独立提交，结果仍以 manifest 为准 |

本轮不提交或推送 `autoresearch` 分支。若某个旧 artifact 的 manifest 记录该分支，只表示历史来源；不能把它当成当前交付分支。

## 证据优先级

发生冲突时按以下顺序判断：

```text
单元 run_manifest.json / matrix_manifest.json
  > 汇总 CSV/JSON/Parquet
  > 当前源码与测试
  > docs/ 中的 active 文档
  > 历史说明、旧论文草稿
```

每个可写入论文的数字至少要能追溯到：dataset、KIR、seed、split、Gate 配置、checkpoint、阈值选择集和源码 commit。

## 当前实验边界

- 主数据集：CLINC150、BANKING77-OOS、StackOverflow。
- Gate-only 主体：KIR、每意图子中心数、Euclidean/对角 Mahalanobis、受控 Baseline、near-OOS、MiniLM 表示诊断。
- 完整 Cascade 修复：先固定 KIR50/seed42，四种 Gate 共用同一套 Router/Expert；不把旧 smoke 结果覆盖掉。
- 完整 Cascade 扩展：固定 KIR50、seed13/42/87 和四种 Gate，共 36 个系统单元，已全部完成；seed13/87 的下游组件由 GPU 训练，seed42 复用已审计模型。
- MOGB：目前只能在官方协议复现成功后进入主表；审计状态不等于性能结果。
- Gate-only 的 OOS F1 不能替代完整系统的 Known macro-F1 或 overall accuracy。
- 代表性 Cascade 中的线性 controlled baseline 读取了旧 sklearn 版本序列化的
  LogisticRegression；当前 `bo` 环境会给出兼容性 warning。它可作受控参考，若要把
  该 baseline 作为最终主结论，应在当前环境重新训练并固定其模型 hash。

## 运行前检查

```bash
git status --short
python -m py_compile src/pipeline/system_pipeline.py tools/eval/eval_system_pipeline_v19.py tools/eval/run_cascade_repair.py
python tools/eval/run_cascade_repair.py
```

GPU 运行需要当前 `bo` 环境和 RTX 5070 兼容的 PyTorch/CUDA；训练和评价命令的实际环境变量以运行记录中的 `LD_LIBRARY_PATH` 为准，不要把本机临时环境当成方法协议。

## 结果入口

完整 Cascade 修复结果：

```text
../artifacts/s2c/outputs/experiments/cascade_repair/gpu_kir50_seed42/
├── repair_manifest.json  # checkpoint hash、训练参数、GPU/PyTorch环境
├── cascade_summary.csv   # 12个单元的论文可读汇总
├── cascade_error_decomposition.csv
├── expert_models/   # 本轮 GPU 重训的 Banking/StackOverflow Expert
└── evaluations/     # 12 个代表性 Gate→Router→Expert 单元
```

Gate-only 和历史研究结果仍在：

```text
../artifacts/s2c/outputs/experiments/
```

先读对应目录的 manifest，再读汇总文件；不要从某个 `predictions.json` 单独推断结论。

完整 KIR50 矩阵结果（独立于 seed42 修复目录，当前已完成 36/36）：

```text
../artifacts/s2c/outputs/experiments/cascade_full/gpu_kir50/
├── component_plan.json
├── gates/gate_manifest.json
├── downstream/              # GPU 训练组件 provenance 与模型
└── evaluations/             # 36 个系统单元；matrix_manifest.json status=complete
```
