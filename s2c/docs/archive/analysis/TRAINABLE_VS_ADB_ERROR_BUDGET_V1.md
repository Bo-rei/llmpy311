# Trainable-K1 与 ADB 逐样本错误预算 V1

更新时间：2026-08-10

本分析只读取已完成 artifact，不训练、不调参、不修改历史结果。范围固定为 StackOverflow、KIR=0.50、seed={42,87,100}。ADB 使用 BERT/TextOIR 兼容合同，Trainable-K1 使用 Known-only MiniLM 合同；因此这里回答的是同一测试样本上的错误预算差异，不是同骨干 SOTA 排名。

## 对齐结论

- 三个 seed 均通过：ADB 独立测试快照 SHA256 与运行 manifest 一致；文本顺序与对应 protocol view 完全一致；Trainable prediction 的 sample_id 顺序与 protocol view 完全一致。
- 每个 seed 对齐 6000 个测试样本，共 18000 条对齐记录；未导出原始文本或逐样本公开预测。

## 错误预算（3 seed 汇总）

| 状态 | Trainable-K1 | ADB | 含义 |
|---|---:|---:|---|
| Known 正确分类 | 41.32% | 40.13% | 已知样本被正确分类 |
| Known 错意图 | 0.52% | 0.26% | 接受为 Known 但意图错误 |
| Known 被拒绝 | 8.16% | 9.61% | Known false rejection |
| OOS 正确拒绝 | 45.91% | 46.24% | OOS 正确判为未知 |
| OOS 误接收 | 4.09% | 3.76% | OOS false acceptance |

## 当前可解释结论

1. Trainable-K1 在这三个 seed 上把 Known 被拒绝从 ADB 的 19.22%（条件于 3000 个 Known 样本）降到 16.32%，但 OOS 误接收从 7.52% 升到 8.18%。因此当前 StackOverflow 证据是“少一些 Known 误拒、略多一些 OOS 误接收”的工作点交换，不是无代价优势。
2. ADB 与 Trainable 的差异同时包含 backbone、训练目标和边界实现差异；本分析不能把差异归因给 MiniLM 训练或 ADB 边界中的单一因素。
3. 逐样本对齐只证明错误预算比较可信，不改变任何阈值、checkpoint、registry 或 test 选择流程；不能把三 seed StackOverflow 结果外推为跨数据集结论。

## 产物

- 机器可读：`results/analysis/archive/analysis/trainable_vs_adb_error_budget_v1/`
- 图表：`figures/archive/analysis/trainable_vs_adb_error_budget_v1/`
- 逐状态转移仅保留聚合计数；原始文本和逐样本预测不进入公开结果。

## 对齐来源

- seed=42: test_sample_ids_sha256=ed50cc34f4a8e69711527e7db4d5bed877b2e768861d442975d1a7b4907b16c6; adb_test_sha256=a488d555cf47920ad8a41ffa5df37dcbf4cb92862bbcbe6f907191a7a6e94c12; aligned=True
- seed=87: test_sample_ids_sha256=53f65be0e59d34bf1a888b93bca28cd91c57b620e14840d33339b37230b27f1f; adb_test_sha256=1fb9ad644d1b42f1167c453ec84c9c98426988315f1f84d26c80d0c4a9e6f7b6; aligned=True
- seed=100: test_sample_ids_sha256=d3314a896b71cb10edb01ba43bef7ace11f1de7e30f9f7352aeae33c2a87e025; adb_test_sha256=c213abf7a6bd1da6fcf26587fa4555761babbe8fa942cb4c9b6fe9caaa60f200; aligned=True
