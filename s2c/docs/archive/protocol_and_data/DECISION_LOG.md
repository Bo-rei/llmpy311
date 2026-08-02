# 研究决策日志

追加式文件；旧决策不删除或改写。

## D1：多中心不再作为普遍假设

- 证据：E2 的 1,650 个配对单元和 E3 的 720 个 partition-control 单元。
- 决策：不再扩大 K/KIR，也不再把 K=2 写成默认有效配置。
- 原因：Banking77 条件性获益，CLINC150 无稳定获益，StackOverflow 明显退化；稳定聚类不等于有效 OOS 边界。

## D2：当前 adaptive-K 证据不足

- 证据：E3 Known-only reliability 特征在数据集之间方向不稳定。
- 决策：不根据测试 OOS 结果拟合 adaptive-K；不启动全量规则搜索。
- 替代：先研究表示空间是否可以在保留局部几何的同时降低 near-OOS collision。

## D3：历史实验与当前协议隔离

- 历史 Frozen/CE/SupCon/CE-Recon、near/far OOS 和 Pipeline 只作为研究依据或迁移控制。
- `protocol_v2_textoir_v1` 的 E0--E3 是当前正式主协议；两者不混写主表。

## D4：完整 Pipeline 暂缓

- 原因：当前最需要验证的是 Gate 表示机制；下游 Router/Expert 质量会掩盖 Gate 改善。
- 决策：R1 pilot 只做 Gate，不启动 ADB、DA-ADB、MOGB 或完整 Pipeline。

## D5：R1 研究方向

- 目标：在 CE-Recon 中加入 Known-only 的 pairwise cosine relation preservation。
- 预期：降低 representation collision、减少 effective-rank collapse，并保留有用的局部邻域结构。
- 失败门：若 3 个数据集的 pilot 未达到预注册成功标准，停止调 beta，转向外部 baseline 和论文收口。

## D6：R1 pilot 条件性支持（2026-07-28）

- 证据：R1 完成 `108/108` 个 Gate 单元；全局 `beta=1.0` 由三个数据集 seed=42 的 Known train/calibration 指标选择。
- 结果：相对 CE-Recon，K=1 OOS F1 在 CLINC150、Banking77、StackOverflow 分别变化 `+0.0035/+0.0215/+0.0255`；平均 ID Recall 变化 `-0.0054`；effective rank、pairwise relation correlation 和 kNN preservation 均提高。
- 限制：Banking77 near-OOS 下降 `-0.0291`，StackOverflow K=2 仍从 `-0.5883` 变为 `-0.5908`。
- 决策：R1 仅获得条件性表示层证据；可生成 R1_full 计划，但不得宣称普遍解决、不得自动运行 R1_full，也不得启动 ADB、DA-ADB、MOGB 或完整 Pipeline。

## D7：R1_full 计划预检通过（2026-07-28）

- 证据：`plans/R1_full_plan.json`，覆盖 3 个数据集、KIR `{0.25,0.50,0.75}`、5 个 seed、3 种表示和 `K={1,2}`，共 135 个表示 cell / 270 个 Gate 单元。
- 决策：将 R1_full 登记为唯一 active 的下一阶段；固定 R1 pilot 选出的全局 `beta=1.0`，每个 representation 只训练一次并由 K=1/2 共享。
- 限制：当前只完成计划和静态预检，尚未冻结 R1_full provenance，也未启动训练；不运行 E2/E3、ADB、DA-ADB、MOGB 或完整 Pipeline。

## D8：R1_full 受控运行启动（2026-07-28）

- 证据：`R1_FULL_PROVENANCE_SNAPSHOT.json` 与 `R1_FULL_CODE_SNAPSHOT.patch`，计划为 135 个表示 cell / 270 个 Gate 单元。
- 决策：启动唯一 active 的 R1_full；每个 dataset/KIR/seed/representation 只训练或编码一次，K=1/2 共享表示，使用固定 `beta=1.0`。
- 限制：只允许读取冻结 E2 cache 和 Known train/calibration；不启动 E4--E7、ADB、DA-ADB、MOGB 或完整 Pipeline。

## D9：R1_full 条件性表示证据收口（2026-07-28）

- 证据：R1_full 完成 `135/135` 表示 cell、`270/270` Gate 单元，失败与无效均为 `0`。
- 结果：Geometry 相对 CE-Recon 的 K=1 OOS F1 在 CLINC150/Banking77/StackOverflow 分别为 `+0.0040/+0.0193/+0.0045`；near-OOS 分别为 `+0.0033/-0.0218/-0.0037`。
- 决策：R1_full 支持“几何保持改善单中心 OOS 的条件性表示层证据”，不支持“near-OOS 已解决”或“多中心普遍有效”；StackOverflow K=2 仍是结构性失败。
- 下一步：先进行论文 claim 审阅和直接 baseline 规划，不扩大 K/KIR，不启动完整 Pipeline。

## D10：R1 StackOverflow K=1/K=2 数值审计（2026-07-28）

- 证据：`R1_FULL_K1_K2_AUDIT.md`、R1 pilot/R1_full 的 `*_k1_k2_comparison.csv`，以及历史
  `results/gate_only/kir_k_fixed_mean_std.csv`。
- 结论：历史 Frozen v19 的 KIR50、对角马氏 K=2−K=1 约为 `-0.0622`；R1 pilot 的
  `-0.5908` 确实是 Geometry CE-Recon 的 combined OOS F1 配对均值，不是误读。R1_full
  中该表示的 15 个单元均值为 `-0.4852`，而 Frozen MiniLM 为 `-0.0915`。
- 决策：不得把历史 Frozen 结果与 R1 的 CE-Recon/Geometry 结果直接合并。该差异应解释为
  表示适配改变了局部中心/半径边界行为；R1 改善 K=1，不代表改善 K=2。
- 限制：`-0.5908` 仍需同时注明 protocol、representation、KIR、distance 和 metric；
  不得脱离这些字段作为一般性 StackOverflow 退化幅度。
- 机制提示：代表性 Geometry 单元的 ID Recall 从 `0.8420` 升至 `0.9577`，而 false
  acceptance 从 `0.0240` 升至 `0.8173`；当前证据更支持多球接受区域并集过覆盖，而非
  Known 碎片化误拒绝是该单元的主因。

## D11：R1 contract repair 收口（2026-07-28）

- 动机：审计发现历史 CE 使用 `head(pooled)`，旧 R1 使用 `head(student_norm)`；
  `pairwise_relation_metrics` 用 teacher distance 计算了 student intra/inter；near/medium/far
  cutpoints 使用了 test OOS quantile。
- 修复：新阶段 `r1_contract_repair_v1` 显式记录 `pooled/pooled_norm/teacher_pooled/teacher_norm`，
  默认分类输入为 pooled，几何关系输入为 normalized_pooled；student 与 teacher 几何统计分开；
  当前 Known-only calibration 没有 validation OOS，near/far 不再从 test 生成。
- 规模：StackOverflow/KIR50/data seed `{42,87,100}`，12 个 trainable checkpoints、30 个 Gate
  单元，0 失败；旧 R1、E0--E3 artifacts 未修改。
- 结果：pooled 与 normalized head 对 K=1 OOS F1 的差异很小（`+0.0002`），对 K=2 差异明显
  （normalized 相对 pooled `-0.0558`）；pooled-head 下 geometry loss 的 K=1 变化仅 `+0.0009`，
  K=2 仍严重退化。修复后的 student geometry 指标不再复用 teacher distances。
- 决策：旧 R1 标记 `completed_but_superseded_by_contract_audit`；旧 Gate predictions 保留，旧
  geometry statistics 标记 `invalid_metric_implementation`，旧 near-OOS 标记
  `exploratory_test_defined_bucket`。本 pilot 不授权 corrected R1_full。
- 下一步：只做 contract-repair 结果的 claim 边界审阅；不自动运行 R1_full、外部 baseline 或完整 Pipeline。

## D12：停止固定 KMeans 多中心救援（2026-07-28）

- 范围：StackOverflow、KIR50、seed `{42,87,100}`，复用 Frozen 与 contract-repair pooled-head
  checkpoint；60 个预注册轻量评分单元，没有 encoder 训练。
- 归因：shared-intent diagonal covariance 相对 per-cluster covariance 明显降低 K=2 false
  acceptance，是唯一一致改善组件；但三种表示均未满足 K=2 相对 K=1 的安全门。按半径归一化选球
  与 Known-train q95 半径进一步扩大接受区域并恶化 OOS F1。
- 决策：`stop_fixed_kmeans_multicenter_rescue`。StackOverflow 的多中心失败不再归因于单一
  classifier contract，也不能通过本轮合理边界替换消除；不再新增损失、K、半径或 adaptive
  selector 来救活该路线。
- 下一步：转入统一协议的最小外部 Baseline pilot；完整 Pipeline 继续等待最终 Gate 候选冻结。

## D13：MiniLM 训练不能救活 StackOverflow 固定多中心（2026-07-28）

- 范围：新独立阶段 `minilm_training_and_stackoverflow_repair_v1`；StackOverflow Frozen K=1/K=2
  逐样本与子簇审计，以及 3 个数据集、KIR=0.50、3 seeds、5 种表示、2 个 K 和 2 种距离的
  `180/180` Gate pilot；训练 checkpoint `36/36`。
- 审计：E2 cache 与 canonical view 的 sample ID、train/calibration/test 顺序和 embedding bytes
  均通过校验；Frozen K=1/K=2 重新得到 E2 的 detector 指标，未发现评分、协方差、半径或缓存
  对齐实现错误。StackOverflow K=2 新增 OOS false accept 由逐样本与子簇表直接记录。
- 结果：Full CE 和 CE-Recon 在部分数据集改善 K=1，但 StackOverflow 的 K=2 OOS F1 相对 K=1
  仍下降约 `0.58--0.60`；SupCon 的下降约 `0.27--0.30`；Frozen/head-only 的下降约
  `0.11--0.14`，均超出预注册安全门，且 false acceptance 同步增加。
- 决策：停止“通过表示训练救活固定后处理多中心”路线；保留最强 K=1 表示对照和 StackOverflow
  作为 boundary-union failure 证据。不扩展 KIR/seed/K 网格，不运行 corrected R1_full、ADB、
  DA-ADB、MOGB 或完整 Pipeline。
- 限制：pilot 没有合法 validation OOS，因此没有把 near/medium/far 分桶纳入正式成功标准；
  Gate-only 结果不能写成完整 Cascade 证据。

## D14：MiniLM pilot closeout 汇总校正（2026-07-28）

- 发现：初版 closeout 表格在按 distance 展示 K=2−K=1 时，取到了最后一个 seed，而不是三个
  seed 的均值；原始逐单元 Gate 结果和 paired delta 文件不受影响。
- 修复：只修正汇总器并重新生成 closeout，明确写入三 seed 均值；没有重跑训练、Gate、审计或
  修改任何 checkpoint/embedding。
- 结论：校正后的 Frozen/head-only、Full CE、SupCon、CE-Recon StackOverflow K=2 平均退化仍分别
  约为 `-0.11--0.14`、`-0.58--0.60`、`-0.27--0.30`、`-0.58--0.60`，停止固定多中心救援的决策不变。

## D15：引入 MOGB 作为隔离的直接基线（2026-07-29）

- 动机：固定 KMeans 多中心路线已经停止，但论文仍需要一个与多粒球/自适应边界直接相关的外部参照。
- 约束：不把 MOGB 改造成 s2c 方法；不覆盖 E2/E3/R1/M1；所有公平模式共享
  `protocol_v2_textoir_v1` registry、views、MiniLM cache 和评价器。
- 实施：固定上游 commit `5b689e2a03de0d86ec41212825e5db8d7f0e5c02`，完成官方代码审计；新增独立
  `protocol_v2.experiments.mogb` 粒球适配层、组件混合模式、单元测试和 StackOverflow smoke。
- 结果：先完成 30/30 pilot，随后按登记计划完成 270/270 MiniLM fair cells（3 数据集、3 KIR、5
  seeds、6 方法）。冻结 MiniLM 上官方式 mean-radius adaptive balls 的 StackOverflow 三 seed OOS F1 约为
  `0.7317`，但 Known Recall 约 `0.2784`；`mogb_partition_ours_boundary` 的 OOS F1 约 `0.7977`，
  但 Known Recall 约 `0.5179`。这些是组件/协议诊断，不是官方 MOGB 论文复现数字。
- 决策：官方 BERT/TextOIR 复现仍标记 `audited_not_reproduced`，因为上游缺少 `utils`、依赖旧 BERT
  栈且数据契约不同；270-cell 结果只作为统一 MiniLM 下的组件比较。不得把 MiniLM 适配版称为
  官方 MOGB，也不得据此宣称 SOTA。

## D16：CLMSG local-scale 假设在 seed13 停止（2026-07-30）

- 范围：`protocol_v2_textoir_v1`、StackOverflow、KIR50、seed13；冻结 MiniLM cache，6,000
  Known proper-train、1,000 Known calibration、6,000 test；完成 KNN、global/class/hybrid
  local-scale 和四个固定 alpha 的 split conformal，共 26 个正式输出。
- 结果：Single-centroid OOS F1 为 `0.6957`。预注册 primary class-conditional Version C
  alpha=0.05 为 `0.3336`，最佳描述性 Version C（class, alpha=0.10）也只有 `0.4975`。
  alpha=0.05 的 Known false rejection 接近目标，但 OOS false acceptance 仍约 `0.79--0.86`。
- 机制：普通 KNN 的 AUROC/AUPR 为 `0.8786/0.8431`，local-scale 后降至约
  `0.676--0.693/0.672--0.702`。support-point scale 的分母偏向稠密局部点；class conditioning
  修复 nearest-label 路径但没有恢复 OOS 排序。共形变换只校准 operating point，不改变排序。
- 决策：触发 `stop_after_seed13`。不运行 seeds42/87，不实现 manifold、entropy、
  cross-conformal 或完整 CLMSG，不把该方法写为论文核心。旧 global-only development precheck
  保留但不进入正式 26 单元统计。

## D17：仅补充 CLMSG 预声明 seed 的确认性证据（2026-07-30）

- 新授权：用户要求在不重复旧矩阵的前提下尽可能增加实验，因此只补 seed `{42,87}` 的固定
  Version A--C 配置，共 52 个确认性输出。
- 冻结项：`k=10`、四个 alpha、global/class/hybrid 支持模式、Frozen MiniLM、数据 split、
  evaluator 与 seed13 完全一致；不得根据 seed13 test 结果改参数。
- 边界：该授权不恢复 manifold、entropy、cross-conformal 或三数据集 full sweep。先判断失败是否
  跨 seed 稳定，再注册普通 KNN coverage--OOS Pareto 这一独立问题。

## D18：KNN k 敏感性收口（2026-07-31）

- 结果：`k={5,10,20,30}` 的 180/180 全协议敏感性完成；`k=5` 的整体 mean OOS F1 为
  `0.6492`，`k=10` 为 `0.6335`，`k=20` 为 `0.5957`，`k=30` 为 `0.5565`。整体相对同 split
  Single-centroid 的 win/tie/loss 为 `2/0/178`。
- 解释：`k=5` 只是四个 k 里相对最不差的配置，其 45 格 W/T/L 仍仅为 `2/0/43`；
  因此 KNN 不是当前协议下的可推广主方法，只保留为非参数参照。
- 决策：`knn_k_sensitivity_v1` 收口为完成态，不再追加新的 KNN k 网格；下一步只使用其 summary
  和 closeout 进行论文/台账汇总。

## D19：MOGB frozen-MiniLM 组件消融登记（2026-07-31）

- 动机：用户要求在不重复旧矩阵的前提下尽可能补充实验；已有 270-cell MOGB fair matrix 仍把
  adaptive partition 与两种边界适配绑在一起，尚未系统隔离 purity、minimum support、distance 和 radius。
- 计划：围绕默认 `purity_get=1.0`、`purity_select=0.9`、`min_get=5`、`min_select=10` 做单因素变更，
  加上缺失的 `euclidean+mean_std` 与 `mahalanobis_diag+mean` 两个边界组合。共 12 个非默认配置、
  45 个固定 protocol cells/config，即 540 个新单元；默认 45-cell 结果只引用，不重复运行。
- 边界：Frozen MiniLM、registry、cache、evaluator 和 test threshold 全部冻结；所有变体完整报告，
  不使用 test OOS 选择 purity、minimum support、distance 或 radius；不宣称这是官方 BERT MOGB 复现。

## D20：MOGB frozen-MiniLM 组件消融收口（2026-07-31）

- 完整性：540/540 个新单元完成，0 missing/invalid/duplicate；复用默认 MOGB、Single-centroid 和
  MOGB-partition/ours-boundary 各 45 个控制单元。所有配置均使用固定 registry/cache/evaluator，
  生成 1,296 条配对显著性记录，未使用 test 选择配置。
- 结果：`purity_get=0.90` 是最佳 partition-only 描述性设置，但相对 Single-centroid OOS F1 仍低
  `0.0391`。`Euclidean + mean+std` 相对默认 mean-radius MOGB 提高 `0.0526`，总体 OOS F1 与
  Single-centroid 近似相同；但 F1-All 与 Known Recall 分别低 `0.1048` 和 `0.2593`。
- 机制：MOGB mean radius 的主要问题是过窄边界导致 Known false rejection；mean+std 能恢复部分
  coverage，但不能恢复完整 Known 分类。diagonal Mahalanobis 未优于 Euclidean，增加极小粒球也
  没有带来有效增益。
- 决策：`mogb_ablation_v1_frozen_ofat` 标记 complete/do_not_repeat；停止邻近 purity、minimum-size、
  distance/radius 扫描。下一步只审查官方 hierarchical representation-learning 单格链路，因为
  当前证据表明 frozen partition 不是论文性能的充分来源。

## D21：补齐同边界 fixed-K 与 adaptive MOGB 直接对照（2026-07-31）

- 缺口：E2 的 fixed K 使用 s2c `mean+std` 边界，而 MOGB adaptive reference 使用 Euclidean
  mean-radius；两者不能单独归因于 partition granularity。
- 计划：固定 frozen MiniLM、Euclidean、mean radius、nearest-ball evaluator，完整报告
  fixed K=1/2/3/4 与 adaptive MOGB。K=2 的 45 格读取既有
  `ours_partition_mogb_boundary`，只新增 K=1/3/4 共 135 格。
- 约束：同一 registry/cache/evaluator；不使用 test 选择 K；不重跑 E2/E3；不改官方 BERT
  blocker 状态。完成 180 格合并与完整性审计后立即停止该阶段。

## D22：动态粒球划分不能单独解释 MOGB 性能（2026-07-31）

- 完整性：fixed K=1/3/4 新跑 135/135，K=2 从既有 fair matrix 逐格校验输入 hash 后复用
  45/45；adaptive `mogb_minilm` reference 45/45；无 missing、invalid、duplicate 或 mixed input。
- 结果：同一 Frozen MiniLM、Euclidean、mean-radius 与 nearest-ball 下，fixed K=1 的总体 OOS F1
  `0.7808`，比 adaptive 高 `0.0469`，45 个配对格全部胜出；fixed K2/3/4 也分别高
  `0.0251/0.0138/0.0143`。adaptive 的 Known Recall 仅 `0.3121`，而 K1 为 `0.5385`。
- 数据集差异：CLINC150 的 15/15 protocol cells、StackOverflow 的 14/15 cells 由 K1 获得
  fixed-K 最佳 OOS F1；Banking77 的最佳 K 分散在 K1/K2/K4，仍不存在统一多中心规则。
- 决策：关闭 frozen-representation partition granularity 扩展，不再扫描 fixed K 或相邻 purity。
  公开 MOGB 结果若显著更强，剩余可检验来源主要是 hierarchical representation learning 与其
  训练/划分交替，而非动态粒球数量本身。
- 唯一下一步：只做一个隔离、可停止的 modernized official-logic representation smoke；成功也不
  自动等同官方论文复现，失败则形成精确 blocker。

## D23：官方 MOGB smoke 的边界与停止判定（2026-07-31）

- StackOverflow/KIR50/seed0 通过外部 modernized runtime 完成一次 one-epoch 工程 smoke，说明
  pinned BERT CE、epoch-end feature bank、原始 GBNR 粒球和 nearest-ball evaluation 可以在不改
  第三方 checkout 的情况下串通；输出只作执行链证据，不作论文性能。
- 原始 legacy path 的 stale-graph/in-place 失败由 detached feature bank + batchwise fixed-centroid
  loss 修复；Banking77 还暴露官方 `cuda:0` 硬编码，compat 层已改为 active device，但当前无可用
  NVIDIA driver，CPU 尝试在长时间模型/特征执行阶段中断且没有指标。
- 决策：官方模式标记 `partial_blocked`，不伪造 Banking77 数字，不把 StackOverflow one-epoch
  数字写成官方复现，不扩展官方 BERT 全矩阵。已有 270-cell MiniLM-fair、540-cell OFAT 和
  180-cell fixed-K 结果继续作为当前 MOGB 组件证据。
- 下一步：完成 provenance/报告/回归审计；只有获得独立可重复 GPU/legacy 环境后，才重新登记
  收敛官方复现。当前不改 s2c Gate、不启动完整 Pipeline。

## D24：官方 MOGB 兼容层收敛但不升级为严格复现（2026-08-01）

- 证据：`mogb_official_converged_v1` 在 StackOverflow 与 Banking77 的 5 个正式 seed 共
  10/10 单元完成；官方格式 F1-All 均值分别为 40.7243 与 19.2843。
- 边界：运行使用 pinned 官方源码、现代化兼容层、本地 BERT 和本地固定快照；旧版缺失
  `utils`、数据契约和原论文 Known 类抽样细节仍未完全恢复。
- 决策：标记为 `complete_non_strict_reproduction`，只作为官方逻辑执行证据；不得称为
  byte-identical/paper reproduction，不与 MiniLM-fair 或历史论文数字混合。

## D25：BRAK pilot 只作为 Known-only 安全负控制（2026-08-01）

- 证据：StackOverflow/KIR50/seeds 42,87,100 的 30 个意图候选均选择 K=1；K>1 的
  calibration union-risk、交叉意图泄漏和 bootstrap instability 均上升，BRAK 与 fixed K1
  的测试结果一致。
- 决策：`brak_v1` complete/do_not_repeat，但不启动三数据集扩展，不宣称已提出可泛化的
  adaptive-K 方法；oracle test-best-K 仅保留为分析上限。

## D26：DCLOOS 缺少外部负样本时保持阻断（2026-08-01）

- 证据：官方 checkout 与实际 source checkout 均已固定且源码可编译，但 `squad_placeh.tsv`
  等官方 open-domain negative corpus 文件缺失；官方流程同时依赖 pseudo OOS、external open-domain
  OOS 和端到端训练。
- 决策：`dcloos_official_unified_v1` 标记 blocked；不得用 protocol test OOS 替代外部语料，
  不生成指标。只有补齐来源、许可证和固定语料后，才能另行登记 smoke。

## D27：本轮外部基线阶段收口（2026-08-01）

- MOGB MiniLM-fair/OFAT/fixed-K、官方兼容层、BRAK 和 DCLOOS preflight 均已独立登记并写入
  `results/final_baselines/summary.csv`；旧 E0--E3/R1/MiniLM 结果保持不可覆盖。
- 唯一下一步：完成状态/日志/注册表/测试的最终一致性审计；不再扩大 MOGB 官方矩阵、BRAK 或
  fixed-K 搜索。论文主表使用同协议 MiniLM-fair 组件，官方兼容结果单列，DCLOOS 明确 blocker。

## D28：保留 DCLOOS 端到端基线（2026-08-01）

- 纠正：DCLOOS（Zhan et al., 2021）是审稿人点名的 fully end-to-end 基线，不能由 ADB/DA-ADB
  替代，也不能从主比较计划中删除。
- 当前状态统一命名为 `blocked_missing_external_negative_data`：官方流程需要伪 OOS 与外部
  open-domain negative corpus（`squad_placeh.tsv`），本地没有该固定语料，因此尚未生成 DCLOOS 指标。
- 决策：MOGB strict single-cell、MOGB 表示上的 BRAK 与 DCLOOS 外部负样本审计并行推进；若补齐
  官方语料和许可证，登记 `DCLOOS-official`；若只能使用有明确来源的数据替代，单独登记
  `DCLOOS-adapted`，不得与 official 结果混称。

## D29：严格 MOGB、BRAK 表示对照与端到端基线边界收口（2026-08-01）

- 严格 MOGB：StackOverflow/KIR50/seed0 的 `official_fixed` 与 `unified_zero` 两个 seed
  契约均完成，结果完全一致但相对论文 Acc/F1-All/F1-U/F1-K 分别低
  `13.5033/19.1398/9.7424/20.0816` 个百分点；判定为 `not_reproduced_strict`，不写成
  SOTA，也不再扩大官方 BERT 矩阵。
- BRAK 表示迁移：在 Frozen MiniLM、MOGB initial BERT、MOGB trained BERT 三种表示上完成
  18 个固定 K/BRAK 汇总单元；trained BERT 只在 2/10 意图选 K=2，绝对指标仍很差。停止
  该表示迁移扩展，BRAK 只作为 Known-only 负控制。
- 端到端基线：DCLOOS 明确保留为审稿人要求的独立 fully end-to-end 方法；缺少
  `squad_placeh.tsv` 外部 open-domain negative corpus 时状态固定为
  `blocked_missing_external_negative_data`，不得用 protocol test OOS 替代，不得用 ADB/DA-ADB
  代替 DCLOOS。
- ADB/DA-ADB：仅完成可运行性审计；当前环境 `transformers.AdamW` 导入失败，未生成指标。若
  后续修复环境，必须新建 ledger 行、固定同一 registry、并与 DCLOOS 单独报告。
- 唯一下一步：保持当前基线汇总和 provenance 冻结，优先获取/确认 DCLOOS 外部负样本及许可证；
  在此之前不启动新的 MOGB、BRAK、adaptive-K 或完整 Pipeline 实验。

## D30：外部基线单格收口（2026-08-01）

- MOGB：StackOverflow/KIR50 与 Banking77/KIR75 的严格单格均已完成，但相对各自论文参考
  明显偏低，统一标记 `not_reproduced_strict`，不扩展官方 BERT 矩阵。
- DCLOOS：官方 Drive 的 `squad.tsv` 已定位并字节一致映射为上游要求的
  `squad_placeh.tsv`；官方 BERT 单格运行约三小时仍未形成最终 metrics，按预设上限停止，
  状态为 `timeout_incomplete`，中间 predictions 明确排除。
- ADB/DA-ADB：在 `textoir-py39`、torch-native AdamW overlay 和本地 safetensors 转换副本下，
  StackOverflow/KIR=.50/seed=0 均完成；F1-open 分别为 89.4712 和 90.8978。两者是现代化
  兼容边界参考，不是 strict protocol_v2 或多 seed SOTA 结果，也不能代替 DCLOOS。
- 唯一下一步：完成 registry、ledger、summary、公开轻量结果和回归检查后冻结本轮外部基线；
  不启动 MOGB/BRAK/adaptive-K 大矩阵或完整 Pipeline。

## D31：DCLOOS 受限预算结果单独登记（2026-08-01）

- reduced-budget DCLOOS 使用已定位的官方 Drive SQuAD 负样本、固定本地 BERT、KIR=.75、seed=888
  完成了上游 test evaluation；训练产生 5,700 条预测。
- 由于复制的上游 `main.py` 缺失 `json` import，最终 raw metrics 序列化失败；指标由完整预测文件独立重算，
  并以 `complete_recovered_intermediate_prediction` 登记，原 `failed` manifest 不覆盖。
- 结果 Accuracy 88.6842、F1-All 90.2629、F1-U/OOS F1 87.0527、F1-K 90.2916；这不是严格默认预算或论文表格复现。
- 决策：保留作为端到端兼容性证据；不把它与 Known-only Gate 方法混列，不扩展 DCLOOS 矩阵，先完成状态与公开汇总校验。
