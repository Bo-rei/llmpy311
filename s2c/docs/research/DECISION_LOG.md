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
