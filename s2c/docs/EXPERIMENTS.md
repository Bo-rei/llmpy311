# s2c 实验结果入口

本文件只定义结果如何分层和读取，不复制完整实验表。完整原始证据在
`../artifacts/s2c/outputs/experiments/`，可提交 GitHub 的轻量快照在
`results/`。

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
