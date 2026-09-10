# H1 MiniLM checkpoint 选择与 full pipeline 汇报

CLINC150 的 validation-selected、CUDA full-pipeline OOS F1 为 **91.64±0.27%**，相对论文 Ours 91.96% 为 **-0.32 pp**；1/3 个 seed 超过。结论：**三 seed 均值仍未超过论文数值参照**。

协议固定为 historical_v19_paper_main / H1 controlled，KIR=.50，seed=13/42/87；不是严格 H0 复现。论文数据/模型链与 H1 存在差异，因此数值差不能全部归因给某一个模块。std 使用 ddof=0。

## 1. 最终配置与论文比较

| Seed | Validation 选中训练方案/epoch | λ / threshold / mode | Val OOS F1 | Test OOS F1 | 相对论文 pp | Known 接受率 | OOS 误接收率 |
|---|---|---|---:|---:|---:|---:|---:|
| 13 | last2_long / 7 | 2.5 / 0.9 / nearest_sphere | 89.56 | 91.37 | -0.59 | 86.44 | 8.00 |
| 42 | last2_long / 9 | 1.5 / 1.05 / nearest_sphere | 89.17 | 91.56 | -0.40 | 84.58 | 6.55 |
| 87 | lora_long / 9 | 1.25 / 1.1 / nearest_sphere | 90.86 | 92.01 | +0.05 | 88.00 | 7.72 |

## 2. Gate OOS 不变，但下游分类是否损失？

| Seed | Gate / pipeline OOS F1 | Gate / pipeline macro F1 | Gate / pipeline Accuracy | Router / Expert error |
|---|---:|---:|---:|---:|
| 13 | 91.37 / 91.37 | 87.07 / 81.48 | 89.40 / 87.16 | 2.93 / 4.32 |
| 42 | 91.56 / 91.56 | 87.22 / 82.20 | 89.69 / 87.76 | 2.52 / 3.42 |
| 87 | 92.01 / 92.01 | 88.18 / 82.07 | 90.22 / 87.82 | 2.88 / 4.65 |

每个 test 样本均核验 Gate 与 pipeline 的 OOS 判决一致；semantic_gate_enabled=False，Router/Expert 不改 OOS 决策。Gate 的多类预测是最近中心 intent，而 pipeline 使用 Router/Expert，macro F1/Accuracy 可能降低，不能称为全链路无损。
macro F1 均在全体测试样本、Known intents+OOS 标签集合上计算。Router error 分母为 accepted Known；Expert error 是相同分母下 domain 正确但 intent 错误的比例。Known 接受率不是 Known 分类正确率。legacy full_pipeline_known_macro_f1 仅用真实 Known 子集，不与全测试样本 Known F1 混用。

原始 checkpoint 九单元逐样本重算、Gate rejection 及各阶段错误数见[自适应中心与指标对账](historical_trainable_adaptive_centers_presentation.md)。历史聚合与当前重算的小幅差异单列保留。

## 3. 训练方式、参数量和 validation 选择

三个新 recipe 均从本地 all-MiniLM-L6-v2 初始化，保留 384→256→384 residual MLP、mean pooling 和 L2 normalize。一轮仅训练 projection 的 warmup 后训练八轮；batch=64、max_length=256、projection lr=2e-4，last2/LoRA lr=2e-5。损失为中心分类 CE 加 intra/inter 项，temperature=.07、权重 1/.1/.1、inter margin=.20。只使用 Known train 做梯度更新。
last2 只更新最后两层和 MLP，不是 LoRA。projection-only 冻结完整骨干。LoRA 使用 PEFT all-linear、r=8、alpha=16、dropout=.1、bias=none，冻结全部原始骨干，只更新 adapter 和 MLP。

| 方案 | 最大可训练参数量（含 MLP） | base 参数训练情况 |
|---|---:|---|
| last2_long | 3,746,944 | 最后两层参与训练 |
| projection_only | 198,016 | 原始骨干全部 frozen |
| lora_long | 535,936 | 原始骨干全部 frozen |

| Seed | 候选 | 选定 epoch | Validation OOS F1 |
|---|---|---:|---:|
| 13 | existing_last2 | -1 | 88.26 |
| 13 | last2_long | 7 | 89.56 |
| 13 | projection_only | 9 | 87.16 |
| 13 | lora_long | 7 | 89.37 |
| 42 | existing_last2 | -1 | 87.81 |
| 42 | last2_long | 9 | 89.17 |
| 42 | projection_only | 9 | 86.64 |
| 42 | lora_long | 9 | 88.55 |
| 87 | existing_last2 | -1 | 89.65 |
| 87 | last2_long | 9 | 90.66 |
| 87 | projection_only | 9 | 88.63 |
| 87 | lora_long | 9 | 90.86 |

每个 epoch 仅由 validation OOS F1 选择 K=1 边界，再按同一指标选 epoch 和 recipe；并列优先较早 epoch/既有 checkpoint。全部 seed/recipe/epoch 选择锁定后才读取 test。Known F1/Accuracy 无硬 guard。未选中 recipe 没有独立 test-confirmed 排名，不能根据 validation 就宣布某种训练方式在 test 上更好。

此前 projection-only 模型类已实现，但 factory 未暴露对应 mode，导致 seed13 last2 完成后失败；入口现已修复并测试。恢复过程复用已有完整 last2/seed13 checkpoint，不覆盖它，不把那次失败当成性能负结果。

## 4. 剩余 score ranking 与 intent 错误

| Seed | Split | 选中 OOS F1 | 固定 score 阈值 oracle | AUROC | 全拒绝 F1 |
|---|---|---:|---:|---:|---:|
| 13 | val | 89.56 | 89.66 | 94.60 | 68.09 |
| 13 | test | 91.37 | 91.51 | 95.19 | 74.29 |
| 42 | val | 89.17 | 89.21 | 94.68 | 68.09 |
| 42 | test | 91.56 | 91.60 | 95.43 | 74.29 |
| 87 | val | 90.86 | 90.91 | 95.41 | 68.09 |
| 87 | test | 92.01 | 92.22 | 96.07 | 74.29 |

三 seed test 固定 score oracle 均值为 **91.78%**，相对当前工作点的阈值余量只有 **0.13 pp**。论文为 91.96%，因此这些 score 的阈值优化仍不足以关闭当前均值差距。seed13/42 的 oracle 也分别低于论文，不能只展示 seed87。

oracle 是固定已选 score 的事后最佳阈值诊断，不是可部署工作点、不参与选参，也不是所有表示的理论上限。若当前训练仍未超过，结论是这些表示训练方案尚不足，不能用继续扩大同一阈值网格代替表示问题分析。

| 残余 OOS intent（test） | 误接收/总样本数（跨 seed） | 边界附近数 |
|---|---:|---:|
| oos | 62/3000 | 51 |
| change_ai_name | 50/90 | 16 |
| pto_used | 50/90 | 8 |
| no | 48/90 | 11 |
| what_is_your_name | 43/60 | 9 |
| todo_list_update | 36/60 | 10 |
| ingredients_list | 32/60 | 12 |
| uber | 22/60 | 10 |
| report_lost_card | 18/60 | 6 |
| change_language | 17/60 | 11 |
| traffic | 17/60 | 10 |
| interest_rate | 16/60 | 10 |

标签为 `oos` 的集合误接收 62/3000（2.07%）；其余被留作 OOS 的 intent 误接收 662/6750（9.81%）。残余错误主要来自留出 intent；例如 change_ai_name、pto_used、no 的误接收率超过一半。这些是类别级错误证据，不能仅凭 intent 名称就断言某个具体 Known/OOS 语义混淆对。

各 seed 的 OOS intent 集不一定相同，表中分母只包含该 intent 在对应 seed 为 OOS 的样本；near-boundary 为 |score/threshold−1|≤.05，只是分数难度代理，不等同人工语义 Near-OOS。

## 5. 自适应中心、失败尝试和保留建议

| 数据集 / 实验线 | OOS F1 均值±std | 论文 Ours | 差值 pp | 超过 seed 数 |
|---|---:|---:|---:|---:|
| CLINC / checkpoint selection | 91.64±0.27 | 91.96 | -0.32 | 1/3 |
| banking77_oos / adaptive overall | 91.89±0.19 | 88.23 | +3.66 | 3/3 |
| stackoverflow / adaptive overall | 89.96±1.50 | 89.71 | +0.25 | 2/3 |

[已完成自适应中心实验](historical_trainable_adaptive_centers_presentation.md)单独报告 fixed/adaptive/overall。CLINC 的 overall OOS F1 为90.99±.78（0/3 seed 超过论文），SO 为89.96±1.50（2/3），BANKING77-OOS 为91.89±.19（3/3）。adaptive 与 fixed 各自验证选参后的 test 均值差为 CLINC +.16、SO −.23、Banking +.32 pp；自适应中心并非普遍更好。统一 K、扩展边界、细阈值和 per-sphere calibration 未稳定解决 CLINC 差距；局部多中心收益取决于数据集和 seed，不存在中心越多越好。
当前 checkpoint 实验与自适应实验是不同的 validation 候选集合，不按两个 test 均值再挑赢家。本轮应保留上表各 seed 的 validation-selected checkpoint/边界；SO 与 Banking 保留已验证的 adaptive overall 工作点。若要把新表示与自适应中心联合比较，需要另行预先确定候选及验证方案。
H0/H1 差异无法凭现有证据定量归因；本轮 H1 的 OOS 差距发生在表示/边界阶段，Router/Expert 另贡献分类损失。多轮复用 validation 属于开发过程，最终强泛化结论仍需要独立留出集。

## 6. 证据与复现

- 聚合结果：`results/analysis/historical_trainable_checkpoint_selection/`；含 candidate_validation、selected_test、ranking_diagnostics、intent_errors 和选择锁。
- 私有 checkpoint：`../artifacts/s2c/runs/historical_trainable_checkpoint_selection/`，旧 artifact 保持不变。
- 新实验：`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python scripts/experiments/run_historical_trainable_checkpoint_selection.py --artifact-root <新目录> --output-root <新目录>`。
- `--resume` 仅复用匹配且完整的训练 recipe；不覆盖不完整训练，也不覆盖已开始的 test confirmation。
- 报告：`python tools/analysis/build_historical_trainable_checkpoint_selection_report.py`，只读取聚合结果，不训练、不推理。
- 原始文本、embedding、逐样本 prediction/score 不公开；未修改 fulltex.tex，未 commit/push。
