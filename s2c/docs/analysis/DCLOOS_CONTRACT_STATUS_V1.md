# DCLOOS 端到端基线合同状态 V1

更新时间：2026-08-10
活动协议：`protocol_v2_textoir_v1`

## 结论先行

DCLOOS 已经完成“官方代码接入、外部负样本接入和隔离 smoke”验证，但还没有形成可以与当前
Known-only MiniLM fair matrix 直接排名的结果。原因不是方法被证明无效，而是 DCLOOS 的训练合同
与当前 registry 不同：它在自己的输入数据上重新随机抽取 Known 类别，并使用 pseudo-OOS 与外部
SQuAD OOS。故其数字只能放在“端到端外部监督参考”层，不能混入当前公平主表。

## 当前比较对象分层

| 层级 | 方法 | 表示/训练 | 当前状态 |
|---|---|---|---|
| 当前公平主表 | `S2C-Trainable-K1` | Known-only MiniLM 后两层+projection，单中心 Gate | 3 数据集×3 KIR×5 seed，已完成 |
| 当前公平组件 | `Frozen/Fold K`、`Random K2`、`MOGB-MiniLM` | 冻结 MiniLM、无外部 OOS | 已完成，63 行主矩阵 |
| MOGB 外部复现 | MOGB BERT 官方逻辑现代兼容单格 | BERT、粒球、最近子中心、平均半径 | StackOverflow KIR=.50/seed0 与 Banking KIR=.75/seed0；未复现论文数字 |
| DCLOOS 外部参考 | DCLOOS adapted | BERT、pseudo-OOS+外部 SQuAD | 既有 reduced 单元和本轮 smoke；不同监督合同 |
| 历史论文 | `fulltex.tex` 的 `Ours` | 完整 Gate–Router–Expert Cascade | 旧协议历史结果，不与当前 Gate-only 排名 |

## 输入合同审计

当前 StackOverflow/KIR=.50/seed=42 protocol export 的 `known_labels.json` 为 10 个 Known intent：

```text
ajax, svn, sharepoint, apache, linq, excel,
oracle, cocoa, visual-studio, spring
```

DCLOOS 上游在同一输入 train.tsv 上再次按 `seed=42` 和 `known_cls_ratio=.50` 随机抽取 5 个类别：

```text
svn, apache, oracle, ajax, spring
```

两者不相同。因此这次运行不是“同 registry 的 DCLOOS”，而是
`dcloos_adapted_protocol_migration`：当前 train/dev/test 文件只作为正样本快照，Known 类别和
OOS 监督仍由 DCLOOS 自己决定。

外部 SQuAD 负样本快照已保存并校验，SHA256：
`f6bf61866c86d3b11565826c3ca1faa00e31f196e0ad9bfd000ec45575fd426e`。

## 本轮运行记录

| artifact | 预算 | 状态 | 结果 |
|---|---:|---|---|
| `dcloos_official_stackoverflow_kir050_seed42_adapted_v1` | 100 epoch | 失败 | 标量 `.item()` 兼容错误 |
| `dcloos_official_stackoverflow_kir050_seed42_adapted_v2` | 100 epoch | 中断 | 单 epoch 约 110 秒，未到最终 metrics；中间预测全部拒为 OOS |
| `dcloos_official_stackoverflow_kir050_seed42_adapted_smoke_v1` | 3 epoch | 失败 | 上游 `annealing_kl` 越界 |
| `dcloos_official_stackoverflow_kir050_seed42_adapted_smoke_v2` | 2 epoch | 失败 | epoch 从 1 开始导致 schedule 少一个完整 epoch |
| `dcloos_official_stackoverflow_kir050_seed42_adapted_smoke_v3` | 1 epoch | 失败 | schedule 仍按 `total_steps+1`，触发越界 |
| `dcloos_official_stackoverflow_kir050_seed42_adapted_smoke_v4` | 1 epoch | 完成 smoke | OOS F1=0，F1-All=8.31%，仅验证链路，不作性能结论 |

最后一次完整 smoke 的产物：

```text
../artifacts/s2c/external/dcloos_official_stackoverflow_kir050_seed42_adapted_smoke_v4/
```

其结果表现为低预算初始模型几乎把所有测试样本判为 Known，说明 1 epoch 不能代表 DCLOOS
收敛性能。运行时 overlay 的修改只包括 torch AdamW、Transformers tokenizer、本地 BERT 路径、
标量转换、显式 metrics/predictions 导出和 schedule 越界保护；第三方源码 checkout 未修改。

## 与既有 DCLOOS reduced 结果的关系

已有 `dcloos_official_oos_kir75_seed888_reduced_v2` 曾产生 OOS F1=87.05%、F1-All=90.26%，但它
使用 KIR=.75、seed=888、不同 Known 抽样和 pseudo/external OOS，不能与当前 StackOverflow/KIR=.50
五 seed fair 行排名。它只能证明外部监督路线在某个兼容预算下可产出有意义数字，不能证明在当前
registry 上超过 S2C 或 MOGB。

## 当前可写入实验结论

1. DCLOOS 官方代码和外部负样本来源已找到，并且运行链路可进入 BERT 训练、验证、测试和结果导出。
2. 当前 DCLOOS 还没有同 Known list、同 seed、同 test split 的公平主结果。
3. 1 epoch smoke 的 OOS F1=0 是预算/收敛诊断，不是 DCLOOS 性能。
4. 在没有固定 Known-list adapter 之前，不能把 DCLOOS 数字与 `S2C-Trainable-K1`、MOGB-Fair、ADB
   并列作 SOTA 排名。

## 唯一后续动作

如果继续做外部基线，应先实现一个只改变 DCLOOS 数据加载的 `known_labels_file` adapter，使其读取
protocol registry 的 Known list，同时保留 pseudo-OOS/external-OOS 的监督差异，并将结果标为
`DCLOOS-adapted-fixed-registry`。在该 adapter 和独立 GPU runtime 通过单格收敛前，不启动 DCLOOS
多 seed 或完整矩阵，也不再扩大 MOGB 复现矩阵。
