# 当前协议 Cascade 错误预算分析 V1

范围：`protocol_v2_textoir_v1`、StackOverflow、KIR=.50、seeds=13/42/87。  
本报告只对已完成的 Gate/Cascade 预测做 sample_id 对齐，不训练、不调参、不修改原始结果。

## 核心结果

| Gate | Gate-only OOS F1 | Cascade OOS F1 | Gate-only F1-All | Cascade F1-All | Gate FA | Cascade FA |
|---|---:|---:|---:|---:|---:|---:|
| Frozen K=1 | 77.29% | 77.29% | 78.60% | 76.55% | 26.54% | 26.54% |
| Trainable K=1 | 86.71% | 86.71% | 85.65% | 83.25% | 11.14% | 11.14% |

表中 FA 的分母是 OOS 测试样本数；FR 的分母是 Known 测试样本数。Cascade 的 OOS 接受/拒绝由 Gate 决定，Expert
只影响被 Gate 接受的 Known 样本分类。因此 Trainable 的主要收益仍来自 Gate 降低 OOS 误接收，而不是
Expert 单独创造了 OOS 信号。

## 机制结论

1. Frozen K=1 的误差预算主要是 OOS 被 Gate 接受；Trainable K=1 显著降低该项。
2. Known 被 Gate 拒绝的比例在两种 Gate 间接近，说明 Trainable 的提升不是靠大幅拒绝 Known。
3. 被 Gate 接受的 Known 样本再交给同一 Expert 后，Trainable 的 Expert 正确率和最终 F1-All 均更好；
   因而当前协议下应把表示/ Gate 改善视为主贡献，下游 Expert 不是主要混淆源。

## 证据文件

- `results/analysis/archive/analysis/cascade_error_budget_v1/per_seed.csv`
- `results/analysis/archive/analysis/cascade_error_budget_v1/transitions.csv`
- `figures/archive/analysis/cascade_error_budget_v1/cascade_error_budget.png`
- `figures/archive/analysis/cascade_error_budget_v1/cascade_gate_to_expert_effect.png`

该分析不能证明跨数据集 Cascade 优势，也不能把当前结果与历史 fulltex、官方 BERT MOGB 或 DCLOOS
外部 OOS 监督直接排名。
