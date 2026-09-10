# corrected-loss BERT-MOGB Known-only 半径覆盖归因 V1

## 实验目标

本实验不重训任何模型，只复用已经完成的 `mogb_corrected_subcentroid_loss_v1`：

- 相同 checkpoint
- 相同 official data snapshot
- 相同 selected-ball 构造逻辑
- 相同欧氏距离与最近粒球判定

唯一改变项是：**只用 dev / calibration Known 样本标定一个全局半径倍率**，检查默认平均半径是否只是工作点过窄。

## 重放验证

- selected ball contract：`fixed checkpoint 下的 deterministic best-effort selected-ball reconstruction`
- selected ball 对比：`{'passed': False, 'reason': 'ball_identity_mismatch'}`
- default 指标最大绝对误差：`1.8667`
- deterministic replay seed：`4`
- source checkpoint SHA256：`127bd347fe9912f81b771630ae8f5d70bd91348567fb8bfd3b2c418da68dfc22`

这说明当前脚本不是重新实现另一套 MOGB，而是在 corrected-loss 运行结果上做半径工作点归因。若 strict replay 失败，本文档不会把当前球结构写成“原 artifact 精确结构复现”。

## Known-only 覆盖工作点

| workpoint | 目标 dev Known coverage | 半径倍率 | 实际 dev Known coverage | F1-U | F1-All | Known Recall | OOS Precision | OOS Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default_mean | - | 1.0000 | 0.5860 | 79.7553 | 71.4203 | 58.6333 | 69.5706 | 93.4333 |
| dev_known_coverage_0.80 | 0.80 | 1.2420 | 0.8000 | 73.4529 | 73.9708 | 77.0000 | 77.7158 | 69.6333 |
| dev_known_coverage_0.85 | 0.85 | 1.3222 | 0.8500 | 66.2415 | 71.6502 | 80.5000 | 78.8046 | 57.1333 |
| dev_known_coverage_0.90 | 0.90 | 1.4351 | 0.9000 | 54.4413 | 68.2563 | 84.2667 | 80.3513 | 41.1667 |
| dev_known_coverage_0.95 | 0.95 | 1.5837 | 0.9500 | 35.6810 | 63.8002 | 87.1667 | 79.9304 | 22.9667 |

## 主要发现

1. 默认平均半径下，corrected-loss MOGB 的 `Known Recall=58.6333`，而把半径扩到 dev Known 95% 覆盖后，Known Recall 提升到 `87.1667`。
2. 这种恢复不是免费的：同一工作点下，测试集 OOS 误接受从 `197` 增加到 `2311`。
3. 因此，corrected-loss MOGB 的性能差距里确实包含**边界工作点过窄**，但只靠放大半径并不能自动变成强 OOS 方法，因为 Known 覆盖恢复的同时会明显扩大 open-space acceptance。

## false acceptance 增量最大的粒球

| ball_id | majority_label | train sample_count | default OOS false accept | cal-95 OOS false accept | delta |
|---|---:|---:|---:|---:|---:|
| 24 | 8 | 13 | 39 | 326 | 287 |
| 12 | 0 | 16 | 11 | 232 | 221 |
| 21 | 6 | 14 | 8 | 215 | 207 |
| 13 | 1 | 36 | 30 | 208 | 178 |
| 23 | 7 | 13 | 12 | 151 | 139 |
| 18 | 5 | 15 | 6 | 142 | 136 |
| 10 | 9 | 14 | 12 | 135 | 123 |
| 9 | 9 | 19 | 6 | 126 | 120 |
| 28 | 0 | 12 | 24 | 139 | 115 |
| 20 | 6 | 24 | 9 | 105 | 96 |

## 解释

这组结果说明，corrected-loss 版本已经比官方 loss 更好，但它仍然存在两个分离的问题：

1. **训练信号问题**：原始官方子中心 loss 压缩了最近子中心信号，这一点此前已被 corrected-loss 实验部分修复。
2. **边界工作点问题**：即使在 corrected-loss 表示上，平均半径仍然偏保守；Known-only 校准能恢复 Known coverage，但会把部分粒球推向更高的 OOS false acceptance。

所以当前最合理的结论不是“只要把 MOGB 半径调大就能复现论文”，而是：

> corrected-loss 修复了部分表示训练问题；但边界仍然需要更稳健的 calibration，否则 default 半径过窄、calibrated 半径又会带来明显的 OOS 误接受扩张。

## 产物

- `results/analysis/archive/analysis/mogb_corrected_radius_coverage_v1/workpoint_metrics.csv`
- `results/analysis/archive/analysis/mogb_corrected_radius_coverage_v1/dev_known_radius_thresholds.csv`
- `results/analysis/archive/analysis/mogb_corrected_radius_coverage_v1/per_ball_radius_attribution.csv`
- `results/analysis/archive/analysis/mogb_corrected_radius_coverage_v1/selected_ball_replay.csv`
- `figures/archive/analysis/mogb_corrected_radius_coverage_v1/radius_workpoints.png`
- `figures/archive/analysis/mogb_corrected_radius_coverage_v1/false_accept_top_balls.png`
