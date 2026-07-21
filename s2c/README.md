# s2c

s2c 是一个开放世界意图识别系统：

```text
文本 → Gate（Known/OOS）→ Router（domain）→ Expert（intent）
```

当前研究问题是：冻结的 MiniLM 语义空间中，Known intent 的局部多簇结构何时能改善 OOS 可分性，以及 Gate 改善能否传递到完整 Cascade。

## 只读这三个入口

1. [项目契约](docs/PROJECT.md)：源码、数据、artifact、证据边界和当前状态。
2. [实验登记与结果](docs/EXPERIMENTS.md)：实验问题、manifest、汇总表和论文可用范围。
3. [运行手册](docs/RUNBOOK.md)：环境检查、预检、复现命令和故障排查。

## 当前结论

- Gate-only 的多簇研究主体已经完成；多簇不是跨数据集普遍有效的默认设置。
- near-OOS 的主要瓶颈是 MiniLM 表示碰撞；局部边界会进一步产生过覆盖。
- BANKING77-OOS 更支持局部多簇，StackOverflow 的固定 KMeans 多球是稳定的负结果。
- Banking77/StackOverflow 的下游 Cascade 已用修复后的 Expert 运行，不应再引用旧 smoke 的低质量数字。
- KIR50、三 seed、四 Gate 的完整 Cascade 为 36/36；MOGB 目前只有协议审计，没有公平性能数字。

## 重要边界

- 当前交付分支是 `main`，本轮不切换、提交或推送 `autoresearch`。
- 大型模型、Parquet 和逐样本结果在 `../artifacts`，不复制进源码仓库。
- 不覆盖既有实验产物；新协议必须使用新的显式 artifact 根目录。
- Gate-only 的 OOS F1 与完整 Cascade 的 Known macro-F1/accuracy 不能混成一张表。
