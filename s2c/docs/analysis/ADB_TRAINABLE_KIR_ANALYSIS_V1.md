# Trainable-K1 与 ADB 的跨 KIR 对比分析 V1

更新时间：2026-08-10

本文只分析已完成的 3 数据集×3 KIR×5 seed 配对结果。Trainable-K1 是当前 Known-only MiniLM 合同，ADB 是 BERT/TextOIR 外部兼容合同；二者共享 protocol split/KIR/seed，但不共享 backbone 和训练实现。因此本文报告同数据工作点差异，不把结果写成同骨干 SOTA 排名。

## 结论摘要

- 训练表示和 ADB 的差异具有数据集/KIR 条件性，不能只看 StackOverflow 单格。
- Trainable-K1 的主要优势集中在 F1-All、Known Recall 和 false rejection；ADB 在部分开放比例下更保守，false acceptance 可能更低。
- 如果只看 OOS F1，会掩盖 Known 覆盖与错误预算差异；因此同时报告 F1-All、Known Recall、false acceptance 和 false rejection。
- 这些结果支持“当前 Trainable-K1 是一个更平衡的同数据工作点”，但不支持跨 backbone 的全面 SOTA 结论。

## 配对统计

所有 delta 定义为 Trainable-K1 − ADB；bootstrap 只用于描述区间，不用于选择参数。

| 指标 | 9 个 dataset×KIR 单元的 delta 均值（pp） | 解释 |
|---|---:|---|
| OOS F1 | +3.11 | 正值表示 Trainable 的 OOS F1 更高；9 个单元正/负=8/1 |
| F1-All | +1.27 | 正值表示整体已知/未知分类更高；9 个单元正/负=6/3 |
| Known Recall | -7.12 | 正值表示 Trainable 保留更多 Known；9 个单元正/负=3/6 |
| False Acceptance | -9.68 | 负值才表示 Trainable 的 OOS 误接受更低；9 个单元正/负=2/7 |
| False Rejection | +7.12 | 负值表示 Trainable 的 Known 误拒绝更少；9 个单元正/负=6/3 |
| Accuracy | +2.52 | 正值表示总体准确率更高；9 个单元正/负=7/2 |

## 解释边界

1. ADB 使用 BERT/TextOIR，Trainable-K1 使用 Known-only MiniLM；因此正向差异不能归因于某一个组件而不考虑 backbone。
2. 训练表示的收益必须和 detector/边界一起解释；已有 fair matrix 显示固定 K>1 在 StackOverflow 上会增加 union false acceptance。
3. 该分析不使用 test OOS 调参，不改变任何 checkpoint、registry、阈值或历史 artifact。

## 产物

- 机器可读结果：`results/analysis/trainable_vs_adb_kir_v1/`
- 图表目录：`figures/trainable_vs_adb_kir_v1`
- 源文件哈希：`{"adb": "0a287bf154bdb0d2db06ec916cfbb4aea40c8b536fabf419942f684e587e1223", "fair": "31ccdd433b38b86ec92b0cd81b460114a7fe1e7f12b94d4a5d031aab7d703145"}`
