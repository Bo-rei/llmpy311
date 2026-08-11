# 当前实验对比图谱 V2（加入 protocol_v2 Cascade bridge）

更新时间：2026-08-10
活动协议：`protocol_v2_textoir_v1`
用途：实验阶段的合同分层、效果比较和机制可视化，不是论文主表，也不做跨合同 SOTA 排名。

## 1. 现在“我的方法”究竟是哪一个

当前自有方法有三个层级，不能混称：

1. **S2C Trainable K=1 Gate**：Known-only 训练 MiniLM 最后两层和残差投影，单中心对角 Mahalanobis Gate。
2. **当前协议 Trainable K=1 Cascade**：上面的 Gate 接入当前 protocol 重新训练的 SmolLM Expert；本轮新增。
3. **fulltex.tex 历史 Ours**：旧合同下的完整 Gate–Router–Expert Cascade，不能与前两者直接排名。

本图谱的当前 fair Gate 主结果中，S2C Trainable K=1 的 OOS F1 为 `87.67%`；
当前协议 Cascade bridge 中为 `86.71%`，Frozen K=1 Cascade 为
`77.29%`。两者分别是 5-seed Gate 矩阵和 3-seed Cascade 桥接，不能混为同一个均值。

## 2. 这次新增的 Cascade 证据

同一 `protocol_v2_textoir_v1` Expert 下，三数据集/KIR=.50/3 seeds 的 Trainable K=1 相对 Frozen K=1
配对差值为：

| 数据集 | OOS F1 | F1-All | Known Recall | false acceptance |
|---|---:|---:|---:|---:|
| CLINC150 | `+1.12pp` | `+1.52pp` | `-1.33pp` | `-2.86pp` |
| Banking77 | `+5.18pp` | `+3.13pp` | `-1.95pp` | `-10.00pp` |
| StackOverflow | `+9.42pp` | `+6.70pp` | `+0.21pp` | `-15.40pp` |

这说明当前 Trainable 表示的收益能传递到下游，而不是只在 Gate-only 指标中出现；StackOverflow 的桥接不是
孤立现象。但它仍然不证明超过完整 MOGB、DCLOOS 或历史 fulltex。逐样本错误预算和每个 seed 的结果见
`docs/analysis/CASCADE_BRIDGE_CROSS_DATASET_V1.md`。

## 3. 当前 MOGB 对比应该怎么读

### 同 MiniLM 公平组件

MOGB-MiniLM 和 MOGB partition + S2C boundary 与当前 S2C fair Gate 使用相同 TEXTOIR split、
Known 列表和 Frozen MiniLM，可用于分析动态粒球、欧氏平均半径和边界的组件作用。StackOverflow/KIR=.50
的五 seed 结果显示：MOGB 组件 false acceptance 较低，但 Known Recall/F1-All 也显著较低；Trainable K=1
是更平衡的工作点。这不是完整官方 MOGB 排名。

### 官方 MOGB 兼容复现

本地运行使用了作者公开代码逻辑（BERT、最近子中心训练、递归粒球、平均半径、最近球推理）和现代兼容层，
但不是作者原始环境的逐字节复现。StackOverflow/KIR=.50/seed=0 的本地结果与论文参考差距，已通过以下证据拆解：

1. 官方子中心损失的距离 L1 归一化压缩了 softmax/logit 梯度；修正后只恢复约 `4.29pp` F1-All。
2. 平均半径过窄造成 Known Recall 约 `51.53%`；Known-only 放大半径后 false acceptance 快速上升。
3. selected-ball 过滤会遗漏部分 Known 类；补球能恢复 Known，但会增加 OOS 误接收。
4. 原始数据快照、Known 列表、环境、最终粒球随机状态和论文训练细节没有全部恢复。

因此正确标签是：`official_code_not_reproduced_under_available_materials`，而不是“已证明 MOGB 算法无效”。

## 4. 外部方法的合同边界

- ADB：StackOverflow/KIR=.50 的 BERT/TextOIR 兼容结果为 `87.36±1.61%`（3 个 CUDA seed），不是同 MiniLM 训练合同。
- DA-ADB：当前兼容运行 NaN/全类预测，标记 invalid，不引用其伪高分。
- DCLOOS：reduced 单元约 `87.05%`，但使用 pseudo-OOS 和外部 SQuAD OOS、KIR=.75/seed=888，不能进入 Known-only fair 排名。
- 历史 fulltex Ours：旧完整 Cascade 的历史主表数字，只能作为旧合同参照。

## 5. 图表和数据

- `figures/comparison_atlas_v2/current_gate_and_cascade_layers.png`
- `figures/comparison_atlas_v2/cascade_bridge_seed_effect.png`
- `figures/comparison_atlas_v2/mogb_paper_gap_attribution_context.png`
- `results/analysis/comparison_atlas_v2/contract_rows.csv`
- `results/analysis/comparison_atlas_v2/MANIFEST.json`

所有输出均为分析层；没有修改 E2/E3/R1/MOGB/DCLOOS 原始 artifact，没有使用测试结果选参。
