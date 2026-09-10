# 跨 KIR 错误预算归因 V1

本分析只读取已经完成的 Trainable-K1 与 MOGB-Fair 五 seed 逐样本转移审计。它不重训模型、不选择阈值、不改变 KIR，也不把测试标签用于方法选择；测试标签仅用于事后统计错误来源。

输入：`/home/bo/bo01/llmpy311/s2c/results/analysis/archive/analysis/trainable_mogb_open_intent_transitions_v1/cell_decomposition_summary.csv`；SHA256：`19eb0d0fa50c07f3106ee3f6afd1b72fb9d86c1a1d23b8a820bd4ed04a1526bf`。覆盖 3 个数据集、3 个 KIR、9 个 dataset×KIR 单元，每单元 5 个 seed。

## 结论

- Trainable-K1 的 F1-All 优势主要来自恢复 MOGB 判为 Known rejected 的正确 Known，而不是增加 OOS 正确拒识。
- 随 KIR 增大，Known 覆盖恢复规模扩大；StackOverflow 的恢复量最大，说明 MOGB-Fair 的平均半径在更高 Known 比例下更保守。
- Trainable 的 OOS 正确净增通常为负，说明它不是通过更激进地拒绝 OOS 获得优势；优势来自更好的 Known 覆盖—边界排序平衡。
- 该结论只适用于当前 MiniLM Known-only 的 MOGB-Fair 组件合同，不能写成超过完整 BERT MOGB 或论文 SOTA。

## 数据集均值（跨三个 KIR）

| 数据集 | F1-All 差值（pp） | Known 正确净增（pp） | OOS 正确净增（pp） | 恢复 MOGB 拒绝 Known（%） |
|---|---:|---:|---:|---:|
| banking77 | 32.48 | 46.28 | -14.11 | 46.52 |
| clinc150 | 35.14 | 41.17 | -2.85 | 41.41 |
| stackoverflow | 43.67 | 56.16 | -6.53 | 56.14 |

## 输出

- `cross_kir_transition_summary.csv`：9 个 dataset×KIR 单元的错误预算指标。
- `figures/archive/analysis/cross_kir_transition_attribution_v1/cross_kir_transition_attribution.png`：四项跨 KIR 热图。
- `figures/archive/analysis/cross_kir_transition_attribution_v1/known_oos_budget_cross_kir.png`：Known/OOS 正确预算折线图。
