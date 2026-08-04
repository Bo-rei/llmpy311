# s2c 项目契约

## 项目边界

s2c 是开放世界意图识别系统，运行链路固定为：

```text
文本 → MiniLM Gate（Known/OOS）→ Router（domain）→ Expert（intent）
```

当前活动代码、配置、测试和文档都在本项目目录。当前正式实验协议是
`protocol_v2_textoir_v1`；E2/E3 已冻结，旧 R1 pilot/full 已通过 contract audit 标记为 superseded，
新的 `r1_contract_repair_v1` 已完成 StackOverflow/KIR50 的 12 个 checkpoint 和 30 个 Gate 单元；
随后 60 单元多中心边界归因已收口并触发停止固定 KMeans 多中心救援；新的
`minilm_training_and_stackoverflow_repair_v1` 又完成了 36 个 MiniLM checkpoint、180 个 Gate
单元和 StackOverflow 逐样本 K=1/K=2 审计，确认表示训练不能稳定修复固定多中心过覆盖。
E3 只比较分簇方式并分析 Known-only 稳定性/覆盖信号，不定义最终 adaptive-K；contract repair
明确区分 classifier input、student/teacher geometry 和 validation-only OOS bucket。MOGB 现已作为
独立外部基线完成源码审计，并分别完成 270-cell MiniLM 公平矩阵与 540/540 frozen-MiniLM OFAT
消融，以及同一 Euclidean+mean-radius 下 180/180 fixed-K 对照；动态粒球没有超过 fixed K1，
后者的 default MOGB、`get_090` 与 Euclidean+mean_std 结论只应视为组件对照，官方
BERT/TextOIR 复现仍 blocked。其结果不修改 E2/E3/R1/M1，也不能直接称为 SOTA。完整 Pipeline 仍需
另行登记。
随后 `clmsg_v1` 完成了 StackOverflow/KIR50 的三-seed confirmation，KNN 又完成
`k={5,10,20,30}` 的 180/180 敏感性矩阵。两条路线均未超过 single centroid，已触发停止门；
manifold、entropy 和 cross-conformal 未启动。

## 工作区职责

| 目录 | 职责 | 是否提交父仓库 |
| --- | --- | --- |
| `s2c/` | 活动源码、配置、测试、文档、轻量公开结果 | 是 |
| `../assets/` | 数据集和基础模型 | 否，保持忽略 |
| `../artifacts/` | 原始实验输出、checkpoint、embedding、逐样本结果 | 否，保持忽略 |
| `../archives/` | 本地历史材料和日志 | 否，保持忽略 |
| `../textoir/` | 独立上游仓库 | 否，保持独立 Git |

`s2c/results/` 只存由 `configs/public_results.yaml` 白名单导出的轻量 CSV/JSON，
不替代 `../artifacts/`，也不包含模型、embedding、checkpoint 或逐样本输出。

## 源码边界

- 当前协议的 data、evaluation、experiments、Gate、runtime 和 tracking 位于
  `src/protocol_v2/`，统一使用 `protocol_v2.*` import。
- 历史 Router、Expert、Gate、pipeline 和严格 SVDD 基线位于 `src/legacy/`，统一使用
  `legacy.*` import；它们只为历史 artifact 和兼容调用保留。
- 训练/评价/导出入口位于 `tools/`；公开快照维护入口位于
  `tools/maintenance/export_public_results.py`。
- `configs/experiment_registry.yaml` 登记原始实验的入口、manifest 和汇总文件；
  `configs/public_results.yaml` 登记允许进入 GitHub 的精确文件。

## 当前活动入口

1. [README.md](../../../README.md)：项目入口。
2. [METHOD.md](../../METHOD.md)：当前方法、源码边界和 adaptive-K 原型。
3. [CURRENT_STATUS.md](../../CURRENT_STATUS.md)：唯一当前状态入口。
4. [EXPERIMENTS.md](../../EXPERIMENTS.md)：结果层次和公开快照。
5. [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md)：审计、测试和复现命令。

历史说明只在 `docs/archive/`，不能覆盖当前事实。读取完整实验数字时，先看
`../artifacts/s2c/runs/protocol_v2_textoir_v1/` 下的 manifest，再看汇总 CSV；读取 GitHub
数字时只看 `results/MANIFEST.csv` 及其对应文件。E3 收口摘要位于
`../artifacts/s2c/runs/protocol_v2_textoir_v1/e3_mechanisms/summaries/`；其中
`E3_partition_paired_effects.csv` 是分簇配对证据，`E3_cluster_stability.csv` 和
`E3_reliability_features.csv` 是诊断证据。R1 收口证据位于
`../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_geometry_preserving_representation/summaries/`，
入口是 `R1_CLOSEOUT.md`；R1 的 beta 选择只使用 Known train/calibration。
旧 R1 的几何字段和 test-defined near-OOS 已被 contract audit 标记为无效/探索性；修复 pilot 的
入口是 `../artifacts/s2c/runs/protocol_v2_textoir_v1/r1_contract_repair_v1/R1_CONTRACT_REPAIR_CLOSEOUT.md`。
多中心边界归因入口是
`../artifacts/s2c/runs/protocol_v2_textoir_v1/multicenter_boundary_attribution/BOUNDARY_ATTRIBUTION_CLOSEOUT.md`。
MOGB 审计和适配入口是 `docs/archive/mogb_reproduction/mogb_integration/`，运行结果根包括
`../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_baseline_v1/`（MiniLM 公平矩阵）与
`../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_ablation_v1/`（frozen-MiniLM OFAT closeout）、
`../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_fixed_k_mean_ablation_v1/`（fixed K1--4 与
adaptive 的同边界对照）。
CLMSG 审计与停止门报告位于 `docs/archive/failed_adaptive_k/clmsg/`，机器可读 closeout 位于
`../artifacts/s2c/runs/protocol_v2_textoir_v1/clmsg_v1/summary/`。
