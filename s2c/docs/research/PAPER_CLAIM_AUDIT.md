# fulltex.tex 论断审计

审计对象：`fulltex.tex` 当前稿。状态不是对论文全文的改写，而是对已有论断的证据审查。

| 原稿论断 | 当前证据 | 状态 | 建议 |
| --- | --- | --- | --- |
| 多簇边界普遍改善 OOS | E2/E3 显示 Banking77 条件性、CLINC150 无稳定收益、StackOverflow 退化 | soften | 改为“多簇收益依赖数据集内部结构与边界匹配” |
| MiniLM 无需训练即可自然形成有用多簇 | Frozen MiniLM 有可读意图结构，但 E3 证明稳定簇不保证边界有效 | soften | 区分“形成局部结构”和“结构适合 OOS 拒识” |
| 方法达到 SOTA | 原稿历史协议表；新协议尚未完成统一外部 baseline | pending | 在统一协议和直接 baseline 完成前删除或限定为历史比较 |
| K=2 有充分依据 | 原稿主要基于局部实验；E2 未支持统一最优 K | soften | 写成历史默认配置，并报告 KIR/K 敏感性 |
| OOS validation 仅用于普通调参 | 原稿第 4.6 节和旧代码存在 OOS validation 使用；当前 protocol_v2 使用 Known-only calibration | soften | 明确历史与当前协议差异，禁止混写 |
| Gate、Router、Expert 共同贡献主结果 | 历史 12/36 Cascade 有结果；当前 E0--E3 只完成 Gate 机制 | pending | 重新训练/审计下游后再宣称新 Gate 的系统收益 |
| Historical results 与当前协议可直接比较 | 数据、split、KIR 定义和结果层次不同 | remove | 所有主表标注 protocol/version，历史结果单列 |
| near-OOS 主要由表示 collision 导致 | 历史 v21/v22 collision 分析支持；当前 R1 将在新 protocol 上验证 | keep_pending | R1 pilot 需要确认 geometry-preserving adaptation 是否改善 collision |

## R1 更新

R1 pilot 与 R1_full 已在当前 protocol 完成（108/108 与 270/270 Gate 单元）。Geometry-Preserving
CE-Recon 相对 CE-Recon 的 K=1 OOS F1 在三个数据集均为正向变化，且有效几何指标改善；但
Banking77/StackOverflow near-OOS 下降，StackOverflow 的 K=2 仍严重退化。因此只能把“几何保持
可条件性缓解单中心表示碰撞”作为 pending/softened claim，不能改回“多中心普遍有效”或“完整
Cascade 已被新方法验证”。直接 baseline 和完整 Pipeline 仍待独立阶段。

## Contract-repair 更新

| 新审计问题 | 当前证据 | 状态 | 建议 |
| --- | --- | --- | --- |
| CE classifier input 是否与历史契约一致 | `R1_CONTRACT_REPAIR_GATE.csv` 明确区分 pooled 与 normalized_pooled；K=1 均值差异很小，K=2 差异明显 | keep_with_scope | 论文必须显式写出 classifier input，不能把旧 R1 与修复 pilot 合并 |
| student intra/inter 是否真实反映 student 几何 | `R1_CONTRACT_REPAIR_GEOMETRY.csv` 同时报告 student 和 teacher 字段；单元测试覆盖不同几何 | keep | 旧 R1 geometry columns 标记 invalid_metric_implementation，修复列才可用于新分析 |
| near/medium/far 是否无 test leakage | 当前 Known-only calibration 没有 validation OOS；30 行均标记 `exploratory_unavailable_validation_oos`，未使用 test quantile | soften | near-OOS 不能进入正式成功标准；旧 test-defined bucket 仅作 exploratory |
| Geometry loss 在 pooled-head 下是否修复 K=2 | pooled-head K=1 仅 `+0.0009`，K=2 仍严重退化 | soften | 仅声称 contract repair 澄清了机制，不声称修复多中心 |
| 是否允许 corrected R1_full | pilot 仅 StackOverflow/KIR50/3 seeds，near-OOS formal contract 未满足 | pending | 不自动扩展；需另行批准并先解决 validation OOS 设计 |

## 多中心边界归因更新

| 论断 | 新证据 | 状态 | 建议 |
| --- | --- | --- | --- |
| StackOverflow K=2 失败主要是 per-cluster covariance 实现问题 | shared-intent covariance 能明显缓解，但 Frozen/CE-Recon/Geometry 均未通过安全门 | soften | 写成 covariance 失配放大退化，但不是充分解释 |
| 使用 `min(d/r)` 选球比原始距离最近球更合理 | 三种表示的 false acceptance 均上升，OOS F1 下降 | remove | 不作为默认改进或论文方法 |
| Known-only q95 半径能稳定校准多球 | q95 使 Known Recall 接近 1，同时 OOS false acceptance 接近饱和 | remove | 仅作为 boundary-union failure 诊断 |
| 固定 KMeans 多中心可通过简单边界修补恢复 | 60/60 单元中没有候选通过预注册停止门 | remove | 将多中心限定为 Banking77 条件性模块和 StackOverflow 负面证据 |

## MiniLM training and StackOverflow repair 更新

| 论断 | 当前证据 | 状态 | 建议 |
| --- | --- | --- | --- |
| 训练 MiniLM 能同时改善 K=1 和固定 K=2 | `MINILM_PILOT_SUMMARY.tsv`：Full CE/CE-Recon 的部分 K=1 提升没有转化为 StackOverflow K=2；SupCon 也未通过安全门 | soften | 仅保留“训练目标影响单中心表示与多中心退化程度”的条件性结论 |
| StackOverflow K=2 退化来自缓存或 detector 实现错误 | `STACKOVERFLOW_AUDIT_CONTRACT.json` 与逐样本表；E2 score 复现、sample ID/embedding hash 对齐 | remove | 将其写成固定后处理多中心的结构性 boundary-union failure |
| Full CE/CE-Recon 是跨数据集统一最优表示 | 180 个 Gate 单元中各数据集、距离和 K 的优劣不一致；StackOverflow K=1 有改善但 K=2 恶化更大 | remove | 报告 dataset-conditional trade-off，不宣称统一最佳 |
| MiniLM training pilot 已验证 near-OOS 改善 | 当前协议 calibration 为 Known-only，没有合法 validation OOS bucket；pilot 未把 test-defined bucket 纳入成功标准 | pending | 不能用本阶段结果宣称 near-OOS 已解决 |
| 新 MiniLM 表示已经验证完整 Cascade 提升 | 本阶段只有 Gate-only，未运行 Router/Expert | pending | 完整系统结论必须另行取得固定下游证据 |

## CLMSG Version A--C 更新

| 论断 | 当前证据 | 状态 | 建议 |
| --- | --- | --- | --- |
| 局部样本支持天然优于中心--半径 Gate | seed13 普通 KNN OOS F1 `0.4520`，Single-centroid `0.6957` | remove | 不将 KNN/local support 写成已验证改进 |
| support-point local scale 改善 OOS 排序 | AUPR 从普通 KNN `0.8431` 降至约 `0.67--0.70` | remove | 将局部尺度偏向稠密点记录为失败机制 |
| split conformal 同时解决 coverage 与 OOS separation | alpha=0.05 的 Known FR 接近目标，但 false acceptance 仍 `0.79--0.86` | soften | 只声称 aggregate Known coverage 校准有效，不声称提升 OOS 分离 |
| CLMSG 值得替换当前论文 Gate | 所有预注册 Version C 模式均低于 Single-centroid，seed13 已触发停止门 | remove | 不启动 manifold/full sweep，不进入主方法 |

## KNN k-sensitivity 更新

| 论断 | 当前证据 | 状态 | 建议 |
| --- | --- | --- | --- |
| 普通 KNN 的失败只是 `k=10` 选得不好 | `knn_k_sensitivity_v1` 的 180 个单元显示 `k={5,10,20,30}` 在九个 dataset×KIR 组均未超过 Single-centroid | remove | 不再把 KNN 家族当作可通过调 k 修复的候选主方法 |
| 更小的 k 可以恢复 StackOverflow 的 KNN OOS 表现 | StackOverflow 三个 KIR 组的最佳描述性 k 仍为 `10`，且均为 `0/0/5` 负于单中心 | remove | 将 StackOverflow 写成普通 KNN 也无法修复的结构性难例 |
| KNN 适合作为下一阶段重点扩展方向 | `k=5` 仅在 CLINC150/Banking77 上略优于 `k=10`，仍稳定落后于 Single-centroid | remove | KNN 保留为协议对齐的 nonparametric baseline，不继续扩展 |

## MOGB frozen-MiniLM 组件消融更新

| 论断 | 当前证据 | 状态 | 建议 |
| --- | --- | --- | --- |
| 动态粒球划分本身足以带来 MOGB 级性能 | 540-cell OFAT 中最佳 partition-only `purity_get=0.90` 仍比 Single-centroid 低 3.91 OOS-F1 点 | remove | 明确区分 frozen partition adapter 与官方 hierarchical representation learning |
| MOGB mean-radius 是当前 MiniLM 空间的合适默认边界 | 默认 mean-radius Known Recall 仅 0.3121；mean+std 提升 OOS F1 5.26 点并恢复部分 coverage | remove | 把 mean-radius 写成强拒绝、低 Known coverage 的直接基线，不作为 s2c 默认 |
| MOGB 动态粒球在相同 Frozen MiniLM 边界下优于固定 K | 同一 Euclidean+mean-radius 下，fixed K1 在 45/45 配对格优于 adaptive，平均 OOS F1 高 4.69 点；K2/3/4 也整体更高 | remove | 将差距归因范围收紧到官方 hierarchical representation learning 尚未验证，而非动态粒球数量本身 |
| 改善 OOS F1 等于改善开放意图分类 | Euclidean mean+std 总体 OOS F1 与单中心近似相同，但 F1-All/Known Recall 低 10.48/25.93 点 | remove | 主表必须同时报告 OOS F1、F1-All、F1-K 和 Known Recall |
| diagonal Mahalanobis 更适合粒球边界 | mean 与 mean+std 两种半径下均落后于 Euclidean | remove | 当前 frozen MiniLM 粒球不再扩 diagonal covariance 网格 |
| 增加更多小粒球会改善 OOS 拒识 | `min_select=5` 将平均球数增至约 193.7，但 OOS F1 无实质提升 | remove | 仅作为“粒度增加不等于开放风险降低”的机制证据 |

## MOGB 官方兼容层、BRAK 与 DCLOOS 更新

| 论断 | 当前证据 | 状态 | 建议 |
| --- | --- | --- | --- |
| 已严格复现 MOGB 论文结果 | 10 个官方逻辑兼容单元（StackOverflow/Banking77 各 5 seed）完成，但使用 modernized runtime、本地快照，且官方旧依赖/数据契约不完整 | soften | 写成“official-logic compatibility evidence”；严格论文复现仍 pending，不与论文数字直接比较 |
| 严格 MOGB 单格已达到论文参考结果 | StackOverflow/KIR50/seed0 的两种 seed 契约都完成，但 Acc/F1-All/F1-U/F1-K 为 `75.1667/68.3502/79.9676/67.1884`，分别低于参考 `13.5033/19.1398/9.7424/20.0816` 个百分点 | remove | 明确写为 `not_reproduced_strict` 的负复现证据，不写成 SOTA |
| 官方 MOGB 兼容结果可作为统一协议 SOTA 主表 | 官方 BERT/旧数据契约与 MiniLM-fair、protocol_v2 split 不同 | remove | 将官方格式 F1-All/Accuracy 单列为外部参考；主表使用同协议方法 |
| BRAK 已证明自适应 K 能稳定优于固定 K | 30 个 StackOverflow Known intent 全选 K=1；K>1 Known-only risk 上升 | remove | BRAK 只作为保守安全负控制，不扩展、不宣称新方法 |
| BRAK 在 MOGB 训练表示上恢复了多中心 | initial BERT 全选 K=1；trained BERT 仅 2/10 intent 选 K=2，且绝对 F1-All 约 `0.0228` | remove | 只报告为表示迁移负控制，不授权新的 adaptive-K 研究 |
| ADB/DA-ADB 已有当前协议性能结果 | 新隔离兼容单格已完成：ADB F1-open `89.4712`，DA-ADB `90.8978`；均为单 seed、modernized TEXTOIR，不是 strict protocol_v2 | soften | 可作为外部边界参考；不得扩写成统一多 seed 或 SOTA 结论，也不能替代 DCLOOS |
| DCLOOS 已完成公平结果 | 默认预算单格为 timeout；另有 reduced-budget 单格完成上游 test evaluation 并从 5,700 条预测恢复指标，但不是严格默认/论文复现 | soften | 保留端到端基线；将 recovery 单独列为兼容性证据，不与 Known-only 方法或论文表格混列 |
| 当前基线覆盖足以宣称统一 SOTA | ADB/DA-ADB 未运行，DCLOOS blocked；MOGB 官方非严格、MiniLM-fair 结果不普遍领先 | remove | 只报告已完成、同协议、可审计的结果与阻断项，不做 SOTA 声明 |
