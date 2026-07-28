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
