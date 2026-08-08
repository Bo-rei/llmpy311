# Trainable K=1 + 预测一致性/证据冲突 Gate 实验报告

更新时间：2026-08-05  
活动协议：`protocol_v2_textoir_v1`  
阶段：`consistency_gate_v1`

## 结论先行

在固定的 RACAL Trainable K=1 表示上，增加确定性表面归一化视图和两次固定 MC-dropout 视图，
并用 Known calibration 选择证据 margin/可容忍冲突数，可以小幅降低 StackOverflow 的 false
acceptance，但没有形成明显的总体性能提升。最稳妥的结果是 `evidence_margin`：OOS F1
相对 Trainable K=1 仅增加约 0.02 个百分点，Known Recall 下降约 0.17 个百分点。

这说明当前瓶颈并不只是“缺少一个一致性阈值”，但 Trainable K=1 + 证据冲突检测比继续增加
多中心更安全，值得作为后续单中心拒识方向的候选控制。

## 1. 实验合同

| 项目 | 设置 |
|---|---|
| 数据集 | StackOverflow |
| KIR | 0.50 |
| seeds | 13、42、87 |
| 表示 | 已冻结 RACAL Trainable K=1 checkpoint |
| 视图 | 原始 eval、2 次固定 MC-dropout、NFKC/空白归一化 |
| 中心 | 每意图一个中心，不增加 K |
| 选择数据 | Known calibration only |
| 测试 OOS | 只用于最终评价 |
| 完成 | 3/3，失败 0 |

## 2. 三 seed 汇总

| Gate | OOS F1 | F1-All | F1-K | Known Recall | False Acceptance | AUROC | AUPR-OOS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Trainable K=1 | 0.8671 ± 0.0079 | 0.8576 ± 0.0032 | 0.8567 ± 0.0029 | 0.8392 ± 0.0032 | 0.1114 ± 0.0165 | 0.9175 ± 0.0087 | 0.8878 ± 0.0202 |
| 严格一致性 | 0.8663 ± 0.0074 | 0.8516 ± 0.0018 | 0.8501 ± 0.0013 | 0.8124 ± 0.0060 | 0.0924 ± 0.0182 | 0.9109 ± 0.0094 | 0.8801 ± 0.0214 |
| 证据 margin | 0.8673 ± 0.0076 | 0.8580 ± 0.0027 | 0.8571 ± 0.0024 | 0.8376 ± 0.0020 | 0.1099 ± 0.0145 | 0.9155 ± 0.0115 | 0.8829 ± 0.0265 |
| 组合 Gate | 0.8674 ± 0.0073 | 0.8566 ± 0.0023 | 0.8555 ± 0.0018 | 0.8319 ± 0.0025 | 0.1060 ± 0.0161 | 0.9115 ± 0.0095 | 0.8807 ± 0.0215 |

相对 Trainable K=1：

- 证据 margin：OOS F1 `+0.02pp`、F1-All `+0.04pp`、Known Recall `-0.17pp`、false acceptance `-0.16pp`；
- 组合 Gate：OOS F1 `+0.03pp`、F1-All `-0.11pp`、Known Recall `-0.73pp`、false acceptance `-0.54pp`；
- 严格一致性虽然 false acceptance 下降约 `1.90pp`，但 Known Recall 下降约 `2.68pp`，不能作为默认方法。

## 3. 解释与限制

1. 选择过程没有读取测试 OOS；每个 seed 的冲突容忍数和 margin 都来自 Known calibration。
2. 证据 margin 的提升很小，三个 seed 并不构成“超过强基线”的证据；应视为受控 pilot。
3. 该阶段不涉及多中心，因此不能证明自适应多中心有效，也没有改变 MOGB 的比较结论。
4. 当前最可靠自有方法仍是 Trainable MiniLM + K=1；一致性/证据冲突可以保留为单中心拒识的候选附加模块。

## 4. 证据路径

- 运行根：`../artifacts/s2c/runs/protocol_v2_textoir_v1/consistency_gate_v1/`
- Provenance：`CONSISTENCY_GATE_PROVENANCE.json`
- 汇总：`CONSISTENCY_GATE_SUMMARY.csv`
- 复现脚本：`scripts/experiments/run_consistency_gate_v1.py`
- 配置：`configs/experiments/protocol_v2_textoir_v1/consistency_gate_v1.yaml`

## 5. 决策

不扩展 KIR、数据集、更多 dropout 视图或新的多中心结构。若继续，应先在同一协议下对
Trainable K=1、evidence margin 和强端到端/边界基线做统一多 seed 对照；不能把本 pilot 的
零点几百分点变化称为 SOTA。
