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
