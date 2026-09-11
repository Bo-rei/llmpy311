# OOS 可视化汇报

更新时间：2026-09-03  
汇报对象：导师/课题组  
阶段汇报实验主线：历史 `v19`，H1 controlled Gate + full pipeline  
对比方法：Frozen MiniLM vs Trainable MiniLM，`K=1`

> 本汇报回答两个相连的问题：Trainable MiniLM 为什么能减少 OOS false acceptance，以及这个 Gate 改善能否传递到完整 `Gate → Router → Expert` pipeline？
>
> 机制图是已有运行结果的 post-hoc 聚合分析；full pipeline 结果来自本次使用现有 Trainable checkpoint 的显式 CUDA 推理，没有重新训练 MiniLM、改阈值或改数据划分，也没有修改 `fulltex.tex`。严格 H0 完整 Cascade 与本 H1 控制线分开处理。

> 最新的旧 archive 数据协议、标准 Banking77、LoRA Gate 及重新训练 Router/Expert 的 seed42 全链路结果，单独见[Pipeline 实验阶段汇报](historical_pipeline_experiment_presentation.md)；本页保留 H1 `banking77_oos` 机制图和旧 full-pipeline 对照，不与新结果混排。

> **Banking 命名锁定：**本页所有 `BANKING77-OOS` 图和数字的实际数据键都是
> `banking77_oos`（25 个 Known intent + 53 个 unknown intent）。`fulltex.tex` 的论文表头把它简称为
> `Banking77`，但那只是展示名称；标准 `banking77`（77 个原始 intent、KIR=.50 为 38 Known）只属于
> `historical_v19_archive` / `protocol_v2_textoir_v1` 的另一条实验线，不属于本页图件。

## 一页结论

Trainable MiniLM 的收益来自 **OOS score 排序发生了有方向的变化**，而不是把 Gate 阈值调得更宽松或更激进：

```text
Known-only 表示适配
        ↓
OOS 与最近 Known 中心的几何关系改变
        ↓
更多困难 OOS 的 normalized score 越过固定的 score=1
        ↓
OOS false acceptance 减少
        ↓
OOS F1 提高
```

在 KIR=`.50` 的三个 seed 均值上，OOS F1 和 false acceptance 为：

| 数据集        | Frozen OOS F1 | Trainable OOS F1 |    Δ F1 | Frozen false acceptance | Trainable false acceptance |      Δ FA |
| ------------- | ------------: | ---------------: | -------: | ----------------------: | -------------------------: | ---------: |
| CLINC150      |        88.02% |           89.50% | +1.47 pp |                   7.06% |                      4.11% |  −2.94 pp |
| StackOverflow |        79.02% |           88.78% | +9.76 pp |                  23.52% |                      7.10% | −16.43 pp |
| BANKING77-OOS |        84.82% |           88.47% | +3.65 pp |                  23.66% |                     16.93% |  −6.73 pp |

最清楚的机制证据是：KIR=`.50`、三个数据集、三个 seed 的配对 OOS 样本中，`2,891` 个由 Trainable 从 Frozen 的误接收修复为正确拒识，只有 `506` 个反向退化；全部 KIR 的 `27` 个 seed-run 中有 `26` 个 OOS F1 提升，`9/9` 个 dataset×KIR 均值为正。

## 1. 统一实验口径

本汇报只使用注册表中的 `historical_protocol_reconciliation_20260826` 对应 H1 controlled v19 运行：

- 数据集：CLINC150、StackOverflow、BANKING77-OOS；
- KIR：`.25`、`.50`、`.75`；
- seed：`13`、`42`、`87`；
- 表示：Frozen `all-MiniLM-L6-v2` 与 Trainable `last2 MiniLM + residual projection`；
- Gate：`K=1`、对角 Mahalanobis、半径 `mean + 1 × std`、`nearest_sphere`；
- 决策：normalized score `>1` 判为 OOS；
- Trainable checkpoint：只由 Known calibration 选择，测试 OOS 不参与训练或选择；
- 图中 Known：只作为必要的 support/coverage 参照，不作为主分析对象。

这里的 H1 controlled v19 是 **历史数据协议下的 Gate 控制线**，不是论文主表中严格 H0 的完整 `Gate → Router → Expert` Cascade。协议差异和 H0 缺失输入见[历史协议与结果对账](HISTORICAL_PROTOCOL_RECONCILIATION_V1.md)。

### 1.1 当前 Trainable MiniLM 的 full pipeline 结果

这次真正补的是：把上一轮已有的 Trainable MiniLM K=1 checkpoint 接入现有 v19 Router/Expert，
在三个数据集、三个 seed 上做完整 `Gate → Router → Expert` 评估，共 `9` 个单元。full pipeline
结果如下；数值为三个 seed 的均值 ± seed 标准差。

| 数据集        | Gate             |       OOS F1 | full-pipeline macro F1 |     Accuracy | Known Recall | OOS false acceptance | Expert error |
| ------------- | ---------------- | -----------: | ---------------------: | -----------: | -----------: | -------------------: | -----------: |
| CLINC150      | Frozen K=1       | 88.02±1.00% |           74.90±1.12% | 83.06±1.16% | 73.67±0.90% |          7.06±1.54% |  4.03±0.57% |
| CLINC150      | Trainable H1 K=1 | 89.50±0.84% |           76.98±1.13% | 84.96±0.99% | 73.44±1.18% |          4.11±1.11% |  3.45±0.52% |
| StackOverflow | Frozen K=1       | 79.02±4.63% |           75.49±2.92% | 76.73±3.79% | 83.22±0.33% |         23.52±7.72% |  7.50±1.70% |
| StackOverflow | Trainable H1 K=1 | 88.78±2.71% |           83.38±1.96% | 85.89±2.21% | 83.71±0.27% |          7.10±4.87% |  5.77±2.17% |
| BANKING77-OOS | Frozen K=1       | 84.82±1.44% |           62.42±1.22% | 76.74±1.58% | 88.83±1.79% |         23.66±2.65% | 12.24±1.24% |
| BANKING77-OOS | Trainable H1 K=1 | 88.47±0.82% |           65.89±1.57% | 81.25±0.88% | 85.47±1.88% |         16.93±1.34% | 11.51±1.26% |

相对 Frozen K=1，Trainable H1 K=1 的 gate 差值为：

| 数据集        | Δ OOS F1 | Δ full macro F1 | Δ Accuracy | Δ Known Recall | Δ OOS false acceptance | Δ Expert error |
| ------------- | --------: | ---------------: | ----------: | --------------: | ----------------------: | --------------: |
| CLINC150      |  +1.47 pp |         +2.08 pp |    +1.90 pp |       −0.24 pp |               −2.94 pp |       −0.58 pp |
| StackOverflow |  +9.76 pp |         +7.89 pp |    +9.17 pp |        +0.49 pp |              −16.43 pp |       −1.73 pp |
| BANKING77-OOS |  +3.65 pp |         +3.47 pp |    +4.51 pp |       −3.37 pp |               −6.73 pp |       −0.73 pp |

这里要正确理解 OOS F1：Router/Expert 只处理被 Gate 接受的 Known 样本，不改变 OOS 的最终标签，
所以 full pipeline 的 OOS F1 与 Gate-only OOS F1 相同是预期现象。新增的端到端证据是 `full-pipeline macro F1`、整体 Accuracy 以及错误路径：CLINC 还有约 `2.36%` Router error，三个数据集的 Expert
error 分别约为 `3.45%`、`5.77%` 和 `11.51%`。因此当前结论是“Trainable Gate 的改善传递到了完整 pipeline”，
不是“Router/Expert 被重新训练后单独变强”。

结果文件：[9 个 Trainable full-pipeline 单元](../../results/analysis/historical_trainable_full_pipeline/per_seed.csv)、
[配对差值](../../results/analysis/full_pipeline_visual_explanation/full_pipeline_paired_effects.csv)、
[错误路径](../../results/analysis/historical_trainable_full_pipeline/error_paths.csv)。本次端到端推理明确使用
`--device cuda`；对应 manifest 记录 `device=cuda`、`test_used_for_selection=false` 和
`oos_used_for_training=false`。

full pipeline 的图只作为结果辅助，不替代上面的数值表：[工作点](../../figures/full_pipeline_visual_explanation/full_pipeline_workpoint.png)、
[错误路径](../../figures/full_pipeline_visual_explanation/full_pipeline_error_paths.png)、
[Gate 到最终 macro F1](../../figures/full_pipeline_visual_explanation/full_pipeline_gate_to_pipeline.png)。

### 1.2 Trainable MiniLM 到底怎么微调？

它**不是 LoRA**。当前实现是“部分解冻 + residual projection”的参数高效微调：

- 基座：`all-MiniLM-L6-v2`，384 维输出；
- 冻结：embedding、前 4 个 Transformer block 和 pooler；
- 解冻：最后 2 个 Transformer block（第 4、5 层）以及 projection；
- projection：`LayerNorm(384) → Linear(384,256) → GELU → Linear(256,384)`，最后一层零初始化，
  以 residual 方式加回 pooled representation，再做 L2 normalization；
- 可训练参数：`3,746,944`；checkpoint 的 `freeze_report` 中没有 LoRA adapter 参数，只有原始最后两层和 projection 参数；
- 优化器：AdamW。warmup 阶段只训练 projection 1 epoch；finetune 阶段同时训练最后两层和 projection，
  projection learning rate 为 `2e-4`，backbone learning rate 为 `2e-5`，最多 3 epoch，patience=1；
- loss：Known train 上的分类交叉熵 + 类内紧致项 + 类间 margin 项，权重分别为 `1.0 / 0.1 / 0.1`，
  temperature=`0.07`，inter margin=`0.20`；
- checkpoint 选择：只在 Known calibration 上用 `f1_k + 0.05 × Known Recall` 选择，OOS 不参与训练或选择。

当前能确认的是：full pipeline 这次使用 GPU 推理；此前历史 Trainable checkpoint 的 legacy manifest 没有记录实际训练 device，
旧 runner 的配置是 `device=auto`，代码在 CUDA 可用时会选择 CUDA，但不能仅凭旧 manifest 把“训练发生在 GPU”说成完全可审计事实。
刚才为补 provenance 启动的显式 CUDA 训练复跑已停止，没有产出新 checkpoint，也没有覆盖旧结果。

## 2. 图件总览

| 图                                                                                                        | 明确问题                                                   | 关键证据                                     | 使用范围                                            |
| --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------- | --------------------------------------------------- |
| [Fig. 1 3D 局部边界](../../figures/historical_oos_visual_explanation/paper_style_local_boundary_geometry_3d.png) | Frozen 最容易误接收 OOS 的局部区域，Trainable 后如何变化？ | Known、OOS、真实 384 维判决、中心和半透明单位球 | StackOverflow / KIR=.50 / seed=42；共同 PCA 的局部白化坐标 |
| [Fig. 1 补充：二维局部边界](../../figures/historical_oos_visual_explanation/paper_style_local_boundary_geometry.png) | 二维投影下的局部边界形状是什么？ | Known、OOS、中心和 score=1 边界              | StackOverflow / KIR=.50 / seed=42；仅作二维投影补充 |
| [Fig. 2 score 分布](../../figures/historical_oos_visual_explanation/paper_style_score_distribution.png)    | OOS 如何跨过固定 Gate 边界？                               | Known、OOS 两条分布与阈值线                  | StackOverflow / KIR=.50 / seed=42；论文 Fig. 4 风格 |
| [Fig. 3 score crossing](../../figures/historical_oos_visual_explanation/paired_oos_score_crossing.png)     | 哪些同一样本被修复或退化？                                 | Frozen score × Trainable score 与两个阈值线 | 三数据集 / KIR=.50 / 三 seed                        |
| [Fig. 4 分解](../../figures/historical_oos_visual_explanation/corrected_oos_score_decomposition.png)       | score 增益来自距离还是半径？                               | 距离 step、带符号半径 step、总 Δscore       | 只看 Trainable-only 修复样本                        |
| [Fig. 5 OOS 密度](../../figures/historical_oos_visual_explanation/cross_dataset_oos_density.png)           | score 变化是否跨数据集成立？                               | OOS-only 三数据集密度                        | KIR=.50 / 三 seed 均值                              |
| [Fig. 6 难度剖面](../../figures/historical_oos_visual_explanation/oos_margin_difficulty_profile.png)       | Frozen 的哪些误接收最容易被修复？                          | 按 Frozen 距离边界的 margin 分层             | KIR=.50 / 三 seed pooled                            |
| [Fig. 7 稳定性](../../figures/historical_oos_visual_explanation/paired_kir_seed_stability.png)             | 提升是否依赖某个 KIR 或 seed？                             | 27 个配对运行、9 个组均值                    | 全部 KIR / 三 seed                                  |
| [Fig. 8 子类型残余风险](../../figures/historical_oos_visual_explanation/oos_subtype_residual_risk.png)     | StackOverflow 哪些 OOS 子类型仍然困难？                    | intent-level false acceptance dumbbell       | KIR=.50；按实际 OOS seed pooled                     |

图中布局参考本地论文 [oos_intent论文.pdf](../../oos_intent论文.pdf) 的 Figure 3（第 7 页）和 Figure 4（第 8 页）：使用 Known 蓝色、OOS 红色、中心星标、虚线边界和低复杂度图例。新图保留论文式视觉语法，同时把主问题收窄为 OOS；validation/test 的协议角色只在文字和源表中保留，不再制造额外图标。

## 3. Fig. 1：3D 局部单位球——真实误接收发生在哪里？

![3D 局部 OOS 边界几何](../../figures/historical_oos_visual_explanation/paper_style_local_boundary_geometry_3d.png)

### 读图方式

- 蓝点：被选中 Known intent 的训练 support；
- 浅玫红色实心点：位于当前展示球外的未知样本；
- 玫红色空心圆：位于当前展示球内的未知样本；
- 深色星：对应的已知意图中心；
- 半透明球面：在该方法局部白化坐标中，归一化 `score=1` 的单位球。

局部区域不是凭视觉挑选的：先在 Frozen 的 OOS false accepts 中找误接收最多的最近中心，再用固定 Frozen score band `[0.90, 1.10]` 观察同一批边界邻域样本。StackOverflow/KIR=`.50`/seed=`42` 中，最高频中心对应 `drupal`，绘图区域保留 `599` 个 Known train support 和 `2,806` 个 OOS。

当前局部球内计数为：Frozen `361 inside / 2,445 outside`；Trainable `5 inside / 2,801 outside`。这正是图中空心点的数量。完整 Gate 的 OOS false acceptance 为 Frozen `361`、Trainable `27`；Trainable 另外 22 个接受样本来自其他 Known 中心，因此不画在当前局部球的空心点中。

### 强解释

Frozen 面板中，当前局部球内的 OOS 以空心圆集中出现；Trainable 面板中，空心圆明显减少。这个图专门解释单个局部支持区域的变化；完整 Gate 的多中心并集结果由源表和其他机制图负责。

### 解释边界

本图现在是保留径向距离的三维示意：方向来自白化表示的 PCA 投影，径向长度重设为该中心的真实 384 维归一化距离。因此三维球内外准确对应所选局部中心的接受边界，但角度和点间距离不能当作原始表示几何。为了让球面和点云同时可读，归一化距离大于 `1.28` 的点沿原方向压缩到显示范围，球内外状态保持不变。图中的 Inside/Outside 只统计 OOS：Frozen 为 361/2445，Trainable 为 5/2801。Trainable 全 Gate 仍误接受 27 个，其中 22 个在当前球外、被其他中心接受，故红色空心圆可能出现在球外。该图不能用于直接比较左右面板的语义位移。二维 PCA 图仍保留为补充。

## 4. Fig. 2：score 分布——OOS 如何跨过固定边界？

![score 分布](../../figures/historical_oos_visual_explanation/paper_style_score_distribution.png)

这是对论文 Figure 4 的简化风格化改造：横轴为 normalized Gate score，黑色虚线是固定的 `score=1`，左侧表示 OOS 被接受，右侧表示 OOS 被拒识。

- 蓝线：Known，用作 coverage guard；
- 红线：OOS，主分析对象；
- 红色浅填充：OOS false-accept 侧。

在 seed=`42` 的 StackOverflow 单元中，OOS false acceptance 从 Frozen 的 `14.6%` 降到 Trainable 的 `2.2%`。关键点是虚线的位置没有变，变化发生在红色 OOS 曲线相对于虚线的位置：Trainable 将更多 OOS 质量推到拒识侧。

这张图不能单独证明泛化，也不能说明所有数据集都呈现同样的单峰平移；它回答的是一个更具体的问题：在同一个工作点上，OOS score 分布是否更容易落在拒识侧。跨数据集和跨 seed 的稳定性由 Fig. 5 和 Fig. 7 补充。

## 5. Fig. 3：配对 score crossing——提升来自哪些样本？

![配对 score crossing](../../figures/historical_oos_visual_explanation/paired_oos_score_crossing.png)

每个点是同一个 OOS 样本的 `(Frozen score, Trainable score)`。两条虚线分别是两个方法共同使用的 `score=1` 边界：

- 左上象限（Frozen<1、Trainable>1）的红点：Trainable-only，Frozen 误接收而 Trainable 正确拒识；
- 右下方向的蓝色空心点：Frozen-only，Trainable 发生退化；
- 灰色 hexbin：全部 OOS 的密度背景，不参与新的估计或选择。

KIR=`.50` 的修复率/退化率为：

| 数据集        | Trainable-only | Frozen-only |   净样本转移 |
| ------------- | -------------: | ----------: | -----------: |
| CLINC150      |           3.5% |        0.6% | 明显偏向修复 |
| StackOverflow |          19.4% |        3.0% |     修复最强 |
| BANKING77-OOS |           8.7% |        2.0% |     偏向修复 |

这张图把 Fig. 2 的分布结论落到了样本层面：StackOverflow 的大幅 F1 增益并不是均值图上的抽象移动，而是大量原本落在阈值左侧的同一样本跨到右侧。反方向的蓝点也必须保留，因为 Trainable 不是对所有 OOS 都改善。

## 6. Fig. 4：精确 score 分解——距离贡献还是半径贡献？

![精确 score 分解](../../figures/historical_oos_visual_explanation/corrected_oos_score_decomposition.png)

只对 `Frozen accepted → Trainable rejected` 的 OOS 做分解。对每个样本，使用同一个 Frozen radius 作为中间基准：

```text
Δscore
= (Trainable distance / Frozen radius − Frozen score)
  + (Trainable score − Trainable distance / Frozen radius)
```

图中：

- 蓝色箭头：距离 step，从 `0` 走到“只替换 Trainable distance”后的 score；
- 橙色箭头：带符号的 radius step，从中间值走到最终 Trainable score；
- 红色菱形：三 seed 均值的总 Δscore；
- 空心圆：三个 seed 的 seed-level 均值。

三个数据集的均值分解为：

| 数据集        | distance step | radius step | total Δscore | 修复样本数 |
| ------------- | ------------: | ----------: | ------------: | ---------: |
| CLINC150      |        +0.144 |     −0.019 |        +0.124 |        341 |
| StackOverflow |        +0.426 |     −0.190 |        +0.236 |      1,744 |
| BANKING77-OOS |        +0.073 |      +0.025 |        +0.098 |        806 |

数字是三 seed 的 seed-level 均值，`n` 是三个 seed 修复样本数之和；因此 `n` 不是独立生物学重复数，也不应被当作一个统计检验样本量。逐样本闭合误差小于 `5.6×10⁻¹⁷`。

### 强结论

CLINC150 和 StackOverflow 的 OOS score 增益主要由 **最近中心距离变大**贡献；StackOverflow 中半径贡献平均为负，但距离贡献更大，仍然得到正的总 Δscore。由此可以排除“Trainable 只是把接受半径整体放大”这一简单解释。

## 7. Fig. 5：跨数据集 OOS 密度——机制是否可迁移？

![跨数据集 OOS 密度](../../figures/historical_oos_visual_explanation/cross_dataset_oos_density.png)

这张图只画 OOS，不把 Known intent 或多种错误类型混进主画面。每个面板先对三个 seed 各自计算 density，再取三 seed 曲线均值；红色浅色区域是 score≤1 的 false-accept 侧。

- CLINC150：Frozen 的 OOS mass 已相对靠右，Trainable 的改善较温和；
- StackOverflow：Frozen 在边界附近的 OOS mass 最明显，Trainable 的右移最明显；
- BANKING77-OOS：分布更宽且有多峰结构，Trainable 仍有右移，但残余低分 OOS 更多。

因此，跨数据集共同成立的是“Trainable 改善 OOS score 排序”，不是“所有数据集都发生同样形状的分布平移”。这也是为什么不能只用一个数据集的漂亮曲线概括方法机制。

## 8. Fig. 6：难度剖面——修复集中在边界附近吗？

![OOS margin 难度剖面](../../figures/historical_oos_visual_explanation/oos_margin_difficulty_profile.png)

将 Frozen false accepts 按 `1 − Frozen score` 分为四个固定区间：`≤0.05`、`0.05–0.10`、`0.10–0.20`、`>0.20`。每根柱子内部显示其中被 Trainable 修复的比例，顶部的 `n` 是该区间的全部 Frozen false accepts。

主要观察：

- StackOverflow 的 `≤0.05` 区间有 `88%` 被修复，`0.05–0.10` 区间有 `44%` 被修复；更深的误接收几乎没有修复；
- CLINC150 和 BANKING77-OOS 也呈现“边界附近修复率高、远离边界修复率低”的结构；
- 这说明 Trainable 的收益主要发生在 **可被表示适配重新排序的边界难例**，而不是任意扩大拒识范围。

该图是对 Fig. 3 的补充：Fig. 3 展示“谁跨界”，Fig. 6 展示“跨界样本原来离边界多远”。

## 9. Fig. 7：KIR/seed 稳定性——是否只是单次运行现象？

![KIR 和 seed 稳定性](../../figures/historical_oos_visual_explanation/paired_kir_seed_stability.png)

横轴是 `Trainable OOS F1 − Frozen OOS F1`，每行是一个 dataset×KIR 组，灰点是三个 seed 的单次结果，红菱形是三 seed 均值，黑色竖线是零增益。

- `26/27` 个 seed-run 的 Δ OOS F1 为正；
- `9/9` 个 dataset×KIR 均值为正；
- 唯一负的 seed-level 例外是 StackOverflow/KIR=`.25`/seed=`87`，约 `−0.20 pp`，幅度很小；
- StackOverflow 的增益随 KIR 变化较大，说明它是最敏感、也最能揭示机制的场景，而不是所有工作点都具有相同的收益幅度。

这张图支持“聚合层面稳定、单次层面并非绝对无例外”的表述。汇报时应保留这个例外，不能说 Trainable 在每一个 seed 上都胜出。

## 10. Fig. 8：StackOverflow 子类型残余风险——哪里仍然难？

![StackOverflow OOS 子类型残余风险](../../figures/historical_oos_visual_explanation/oos_subtype_residual_risk.png)

每行是一个实际作为 OOS 出现过的 StackOverflow intent；蓝点为 Frozen false-accept rate，红点为 Trainable false-accept rate，连线展示变化方向。该图按残余 Trainable 风险排序，并在图中直接注明：红点向左是修复，向右是退化。

这张图提供两个同时成立的结论：

1. 大多数子类型的红点明显向左，说明收益不是只来自一个 intent；`drupal`、`svn`、`visual-studio`、`magento` 等子类型的误接收下降尤其明显。
2. 训练后仍有残余甚至退化区域：`cocoa` 与 `hibernate` 的 pooled false-accept rate 高于 Frozen。它们是后续重点分析对象，说明 Trainable 表示适配不是对所有语义碰撞都有效。

子类型图是 post-hoc 机制诊断，不是新的 checkpoint/阈值选择依据；不同 seed 的 OOS intent 集合可能不同，图下方已注明 pooled run-observation，源表保留每个 intent 的 seed coverage 和计数。

## 11. 这八张图合起来能证明什么？

### 可以支持的强结论

- 在历史 H1 controlled v19、K=1、固定 `score=1` 的合同下，Trainable MiniLM 在三个数据集的 KIR=`.50` 均值上都提高 OOS F1，并降低 false acceptance；
- OOS 性能改善可以追溯到同一样本的 score crossing，而不是只有总体均值变化；
- 修复样本的 score 增益主要由最近 Known center 的距离项驱动，半径项并非主要来源；
- 收益在三数据集、三 KIR、三 seed 上具有聚合稳定性，但存在明确的单次 seed 例外和子类型残余风险；
- 当前机制最准确的表述是“Known-only 表示适配改善了 OOS score ordering / separation”，不是“所有 OOS 都被推远”或“训练后形成完美线性可分空间”。

### 不能从这八张图推出的结论

- 不能说已经完整复现论文严格 H0 主 Cascade；
- 不能把 H1 Gate-only 结果直接填回 `fulltex.tex` 的论文主表；
- 不能把 PCA 中二维点的左右/上下位置直接当成 384 维 Gate 安全性；
- 不能说 Trainable 在所有 seed、所有 OOS subtype 上都优于 Frozen；
- 不能把子类型或测试 OOS 分布用于重新选择阈值、K、checkpoint 或训练超参数；
- 不能把这组 H1 结果与 protocol_v2、BERT/TextOIR 或不同监督合同的结果混成一个无条件 SOTA 排名。

## 12. 汇报建议：按一条证据链讲

建议现场按以下顺序，不要先从 PCA 图开始：

1. **结果**：先用 KIR=`.50` 表格说明三个数据集 OOS F1 都升、false acceptance 都降；
2. **决策**：用 Fig. 2 说明阈值没有变，OOS score mass 变得更常落在拒识侧；
3. **样本**：用 Fig. 3 说明 StackOverflow 的大收益来自大量具体样本跨过 `score=1`；
4. **来源**：用 Fig. 4 说明跨界的主要来源是距离项，不是简单放大半径；
5. **范围**：用 Fig. 5 和 Fig. 6 说明机制跨数据集成立，但主要修复边界难例；
6. **稳健性**：用 Fig. 7 交代 26/27 的 seed 结果和唯一负例；
7. **剩余问题**：用 Fig. 8 指出 cocoa/hibernate 等残余 hard subtype；
8. **限定**：最后强调 H1 Gate-only 边界，不把它说成严格 H0 论文复现。

## 13. 数据、脚本与 QA 入口

分析 bundle：[historical_oos_visual_explanation](../../results/analysis/historical_oos_visual_explanation/MANIFEST.json)

主要源表：

- [KIR=.50 性能聚合](../../results/analysis/historical_oos_visual_explanation/kir50_performance_summary.csv)
- [27 个配对运行稳定性](../../results/analysis/historical_oos_visual_explanation/paired_kir_seed_stability.csv)
- [score crossing 聚合](../../results/analysis/historical_oos_visual_explanation/score_crossing_summary.csv)
- [修复样本 score 分解](../../results/analysis/historical_oos_visual_explanation/corrected_score_decomposition.csv)
- [margin 难度剖面](../../results/analysis/historical_oos_visual_explanation/oos_margin_difficulty_profile.csv)
- [StackOverflow 子类型源表](../../results/analysis/historical_oos_visual_explanation/oos_subtype_residual_risk.csv)
- [重编码一致性检查](../../results/analysis/historical_oos_visual_explanation/stack_overflow_reencoding_verification.csv)

构建脚本：[build_historical_oos_visual_explanation.py](../../tools/analysis/build_historical_oos_visual_explanation.py)

输出格式：每张图均提供 PNG（300 dpi）、TIFF（600 dpi）、PDF 和 SVG。PNG 适合汇报预览，PDF/SVG 适合论文编辑，TIFF 适合期刊栅格输出。

QA 记录：

- Python figure source preflight：14/14 PASS，0 WARN，0 FAIL；
- score 重新编码与历史 prediction：Frozen/Trainable 各 `5,990` 行，score 最大绝对误差分别约 `2.3×10⁻⁷`、`4.3×10⁻⁷`，决策 mismatch=`0`，nearest-cluster mismatch=`0`；
- score decomposition 最大闭合误差：`5.6×10⁻¹⁷`；
- 视觉复核最终 verdict：`91/100`，超过 90 分交付阈值；
- 统计性质：这是描述性分析，没有 p-value、显著性检验或新的 test-time 选择。

旧版 v3 dashboard 和训练曲线仍保留在 [historical_protocol_oos_v3](../../figures/historical_protocol_oos_v3/) 作为历史支撑；本汇报将它们降为 supporting evidence，主路线使用上面的八张 OOS-first 图。
