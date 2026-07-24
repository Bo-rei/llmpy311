# s2c 实验结果入口

本文件只定义结果如何分层和读取，不复制完整实验表。完整原始证据在
`../artifacts/s2c/outputs/experiments/`，可提交 GitHub 的轻量快照在
`results/`。

## protocol_v2（当前：暂停）

`protocol_v2` TEXTOIR candidate 的训练、embedding、Gate sweep、外部方法复现与公平比较
声明继续暂停。它保留为审计对象，`../artifacts/s2c/runs/protocol_v2/` 不覆盖 v19–v22，
也不得作为正式论文结果。

`protocol_v2_official_v1` 已从官方 raw source 重建 CLINC150 与 Banking77。官方 E1 Gate-only
子矩阵已完成：两个已准入数据集、KIR `{.25,.50,.75}`、seed 42、K `{1,2}`、欧氏/对角马氏距离，
共 **24/24** 固定边界单元。每个单元使用同一 frozen MiniLM、本地 canonical embedding cache 和
Known-only 边界；所有 primary Gate 指标均为有限值。轻量汇总在
`results/gate_only/protocol_v2_official_e1_admitted.csv`，原始 manifest/metrics 在
`../artifacts/s2c/runs/protocol_v2_official_v1/`。

这不是原候选 36-cell 三数据集 smoke：StackOverflow 和 BANKING77-OOS 继续 blocked，不能用旧
candidate 数字补齐。E1 也仅是单 seed 的 Gate-only 可运行性证据，不能写成完整 Cascade、三 seed
稳定性结论或论文主表。后续只可为已准入数据集按需物化 view/export 并在独立 run root 登记任务；
禁止 resume 当前候选数据上的 dense/boundary/E4 任务。运行与审计入口见
`docs/DATASETS.md`、`docs/DATA_PROTOCOL_V2.md` 和
`../artifacts/s2c/reports/data_provenance_audit/2026-07-22_three_way_verification/`。

## 两类评价必须分开

### Gate-only

只评价 MiniLM Gate，回答 Known 多簇结构、near-OOS、表示碰撞和局部支持边界问题。
主要指标包括 OOS F1、AUROC、AUPR-OOS、FPR95、ID Recall、representation collision
和 boundary overcoverage。

Frozen、CE、SupCon 及其表示探针结果属于这一层，不能写成完整 Pipeline 结果。

### 完整 Cascade

评价固定的 `Gate → Router → Expert`，回答 Gate 差异能否传递到最终系统。主要指标包括
OOS F1、Known macro-F1、overall accuracy、ID Recall，以及 Gate、Router、Expert
错误阶段。

CE-Recon 同时有 Gate-only 证据和完整 Cascade 证据；两类数字仍必须分表。

## E4：外部 Baseline 的可运行性边界（仅实现审计，未获数据准入）

`configs/experiments/protocol_v2/external_baselines.yaml` 只声明
`CLINC150/KIR=0.50/seed=0` 的 smoke 候选。所有候选都先读取同一个 registry 和固定 views，
不允许外部方法自行抽取 Known intents。

- `msp`、`energy`、`knn`、`lof` 是使用本地 frozen MiniLM 的可控 Gate-only 方法；它们只用
  Known train 拟合、Known calibration 选择阈值，测试 OOS 不参与模型或阈值选择。
- `doc`、`adb`、`da_adb` 只生成 TEXTOIR-format fixed-split export。若没有经过审计的独立
  上游环境与 prediction importer，运行器写 `blocked` manifest，不生成指标。
- `mogb` 只生成 MOGB-format fixed-split export。未配置官方 MOGB 环境时同样写 `blocked`，
  禁止把 MiniLM 的自适应边界替代品称作 MOGB。
- `k_plus_1_way` 的 export 仍是 Known-only train/dev；在不引入真实或合成 OOS 训练样本的
  当前协议下写 `unsupported`，不伪造 K+1-way 数字。

预检不会运行模型：

```bash
python -m s2c.experiments.external_baselines \
  --config configs/experiments/protocol_v2/external_baselines.yaml --smoke
```

只有未来获得数据准入后，显式传入 `--execute` 才能执行可用的 native control，或为预期不可用的上游方法保存
auditable `blocked.json`：

```bash
python -m s2c.experiments.external_baselines \
  --config configs/experiments/protocol_v2/external_baselines.yaml \
  --smoke --method msp --execute --resume
```

E4 的目录为 `../artifacts/s2c/runs/protocol_v2/external_baselines/`，与 E1–E3 Gate run
目录完全分离。被阻塞或不支持的 manifest 不是失败指标，也绝不能与 completed metrics 混合汇总。

## 结果来源

| 用途 | 入口 | 说明 |
| --- | --- | --- |
| GitHub 公开阅读 | `results/README.md`、`results/MANIFEST.csv` | 只含白名单轻量文件 |
| 实验完整审计 | `configs/experiment_registry.yaml` | 检查入口、manifest、summary 和 unit count |
| 原始数字与逐单元证据 | `../artifacts/s2c/outputs/experiments/` | 不提交、不重命名、不覆盖 |
| 研究历史 | `docs/archive/` | 仅作背景，不作为当前结论 |

公开结果导出配置是 `configs/public_results.yaml`；执行：

```bash
python tools/maintenance/export_public_results.py --dry-run
python tools/maintenance/export_public_results.py --execute
python tools/maintenance/export_public_results.py --verify
```

## 当前结果边界

- 公开快照只保留汇总 CSV、协议 JSON 和 provenance，不包含模型或逐样本 scores。
- Gate-only 与完整 Cascade 不合并成单一 F1。
- MOGB 只有协议审计时，公开目录只放审计 JSON，不生成伪性能表。
- 原始实验目录名（包括 v19、v20、v21、v22）属于血缘，禁止移动或重命名。
- 新的公开导出必须更新 `public_results.yaml`，不能模糊复制整个实验目录。
