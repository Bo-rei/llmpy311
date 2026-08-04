# RC-AMBL StackOverflow pilot 实验报告

## 1. 结论先行

本轮在 `protocol_v2_textoir_v1` 下完成了新的 `adaptive_v1 / RC-AMBL` 风险校准自适应多中心边界实验。范围严格限定为 StackOverflow、KIR=0.50、正式 seed `13/42/87`，没有重跑 E2、E3、R1、BRAK 或 MOGB 旧矩阵。

RC-AMBL 的两种模式均完成 `3/3` seed，合计 `6` 个方法单元，失败、缺失、重复和无效指标均为 0；测试 OOS 没有参与中心数、半径、阈值或 margin 选择。但三次候选分裂都被 Known-only 安全门拒绝，最终每个 intent 都回退为 `K_y=1`。因此本轮得到的是“安全回退与风险诊断”，不是多中心成功，也不允许扩展到其他数据集或 KIR。

更关键的是：RC-AMBL 的类级 evidence 阈值虽然把 Known Recall 提高到 `0.9221`，却造成平均 false acceptance `0.5569`，相对 E2 K=1 的 `0.2654` 增加约 `29.14` 个百分点；OOS F1 只有 `0.5785`，比 E2 K=1 的 `0.7729` 低约 `19.44` 个百分点。因此，当前瓶颈不是“没有找到足够多中心”，而是类级证据和 Known-only 阈值在 StackOverflow 上没有形成可靠的 OOS 排序/校准。

## 2. 数据、协议和防泄漏

| 项目 | 固定值 |
|---|---|
| 活动协议 | `protocol_v2_textoir_v1` |
| 数据集 / KIR | StackOverflow / `0.50` |
| seed | `13, 42, 87` |
| 表示 | 冻结 `all-MiniLM-L6-v2`，复用 canonical embedding cache |
| 训练 Known | 6000 行 |
| calibration Known | 1000 行，按 intent 用 `SeedSequence` 拆为 select/threshold |
| 测试 | 6000 行，其中 Known 3000、held-out OOS 3000 |
| 距离 / 半径 | diagonal Mahalanobis / `mean + 1.0 × std` |
| 测试选择 | `false`；未读取测试标签用于任何选择 |

`calibration_select` 只用于候选分裂安全门，`calibration_threshold` 只用于最终 energy、parent score 和 top-two gap 阈值。ProxyOOS 模式只使用 Known calibration 中的留出 intent-pair episode，不使用 test OOS；由于所有分裂都被拒绝，两种模式最后结构和指标相同。

## 3. RC-AMBL 实现

RC-AMBL 每个 Known intent 从一个父中心开始，每轮只尝试一个 PCA 主方向 median split；子簇下限为 `max(10, ceil(0.05*n_y))`，上限为每个 intent 4 个中心。每个局部中心使用收缩对角协方差：

```text
Sigma_local = rho * Sigma_cluster + (1-rho) * Sigma_class + epsilon I
q_yk(z) = d_yk(z)^2 / r_yk^2
E_y(z) = -log sum_k w_yk exp(-0.5 q_yk(z))
```

最终接受 Known 必须同时满足：最优类 evidence 不超过 `tau`、没有越过该类父边界、top-1/top-2 evidence gap 不小于 `delta`。所有候选分裂还必须满足 Known Recall、ambiguity、bootstrap stability、紧致度和复杂度安全门。

## 4. 新方法结果（3 seed 均值 ± 标准差）

| 方法 | OOS F1 | F1-All | F1-K | Accuracy | Known Recall | False Accept | False Reject | AUROC | AUPR-OOS | 平均 K_y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RC-AMBL-KnownOnly | 0.5785 ± 0.0926 | 0.7691 ± 0.0072 | 0.7882 ± 0.0019 | 0.6460 ± 0.0443 | 0.9221 ± 0.0028 | 0.5569 ± 0.1026 | 0.0779 ± 0.0028 | 0.8800 ± 0.0113 | 0.8173 ± 0.0127 | 1.00 |
| RC-AMBL-ProxyOOS | 0.5785 ± 0.0926 | 0.7691 ± 0.0072 | 0.7882 ± 0.0019 | 0.6460 ± 0.0443 | 0.9221 ± 0.0028 | 0.5569 ± 0.1026 | 0.0779 ± 0.0028 | 0.8800 ± 0.0113 | 0.8173 ± 0.0127 | 1.00 |

逐 seed 的 OOS F1（KnownOnly/ProxyOOS 相同）为：seed13 `0.5044`、seed42 `0.6823`、seed87 `0.5487`。方向一致地说明当前 evidence 阈值在该数据集上误接受 OOS 较多；不是某一个 seed 的偶然崩溃。

## 5. 候选分裂诊断

| seed | 被选 intent | 子簇规模 | 紧致度收益 | 稳定性 median ARI | Known Recall delta | 决策 |
|---:|---|---|---:|---:|---:|---|
| 13 | sharepoint | 300 / 300 | 0.0304 | 0.7565 | -0.0080 | 拒绝：稳定性与安全门 |
| 42 | sharepoint | 300 / 300 | 0.0304 | 0.7712 | +0.0080 | 拒绝：稳定性与安全门 |
| 87 | drupal | 300 / 300 | 0.0299 | 0.7051 | -0.0020 | 拒绝：稳定性与安全门 |

三次候选均满足样本量和正紧致度收益，但 bootstrap median ARI 都低于预注册 `0.80`；seed13/87 还出现 Known Recall 下降。因而最终 `K_y` 在 10 个 Known intent 上均为 1。这个结果支持“结构自适应安全回退”确实生效，但不支持 StackOverflow 存在可安全利用的局部多模态结构。

## 6. 与冻结 E2 / 复用基线的关系

以下行来自 hash 校验后的历史 artifact，只作为同协议参照，未被本轮修改：

| 方法 | OOS F1 | F1-All | Accuracy | Known Recall | False Accept | 说明 |
|---|---:|---:|---:|---:|---:|---|
| s2c K=1 nearest-sphere | 0.7729 ± 0.0510 | 0.7860 ± 0.0175 | 0.7530 ± 0.0348 | 0.8371 | 0.2654 | E2 精确复用 |
| s2c fixed K=2 | 0.6715 ± 0.0314 | 0.7359 ± 0.0208 | 0.6816 ± 0.0230 | 0.8766 | 0.4317 | E2 精确复用，union-risk 负面参照 |
| random-balanced K=2 | 0.7713 ± 0.0528 | 未提供 | 未提供 | 0.8408 | 0.2700 | E3 精确复用，F1-All/Accuracy 未在该摘要中提供 |
| BRAK Known-only | 0.8001 ± 0.0277 | 0.7914 ± 0.0209 | 0.7731 ± 0.0022 | 0.8380 | 0.2247 | 仅 2 个正式 seed 行，不能当五 seed 主结果 |

相对 E2 K=1，RC-AMBL 的 F1-All 下降约 `1.69 pp`，F1-K 略高约 `0.09 pp`，Known Recall 提高约 `8.50 pp`，但 false acceptance 增加约 `29.14 pp`。这明确表明只看 Known 覆盖或 F1-K 会掩盖 OOS 拒识失败。

## 7. MOGB 边界

当前主表中的 `MOGB-MiniLM-compatible-component-reference` 不是 MOGB 官方 BERT 复现，而是冻结 MiniLM、统一 split 下的兼容组件参考：OOS F1 `0.7319 ± 0.0009`，Known Recall 只有 `0.2834 ± 0.0055`，F1-All `0.4515 ± 0.0062`。它不应被写成严格 MOGB 或 SOTA 结果。

MOGB 官方 BERT 严格单格也已单独记录：StackOverflow KIR=.50 seed0 的 F1-All `68.3502`、F1-U `79.9676`；Banking77 KIR=.75 seed0 的 F1-All `59.1627`。两者均与论文公开数字差距较大，状态为 `not_reproduced_strict`。详细来源见 `docs/对比实验/MOGB_DCLOOS_对比结果报告.md` 和 MOGB archive，不与本轮 RC-AMBL 数字混合。

## 8. 晋级判定

预注册门要求 RC-AMBL 相对 K=1 的 OOS F1、Known F1、F1-All 不下降超过 1 pp，Known Recall 不下降超过 1.5 pp，且不得出现 false acceptance 爆炸；本轮不满足：

- OOS F1：`-19.44 pp`；
- F1-All：`-1.69 pp`；
- false acceptance：`+29.14 pp`；
- 三 seed 均无 accepted split，`K_y>1` 未出现。

因此判定：`stop_adaptive_v1_pilot`。不能进入 CLINC150、Banking77、其他 KIR、轻量表示适配器或更大 K 网格，也不能把 RC-AMBL 称为 SOTA 或最终方法。

## 9. 下一步

当前唯一合理的后续是先修复“同一 RC-AMBL evidence 合同下的 K=1 校准控制”：明确父边界、energy、gap 的阈值目标与 OOS 排序之间的关系，使用 Known-only calibration 做独立控制实验；如果仍然出现高 false acceptance，则停止 RC-AMBL，保留 E2 K=1 作为 StackOverflow 主结果。不能使用 test OOS 调整阈值，也不能用本轮失败结果授权扩大实验。

## 10. 证据入口

- 运行 artifact：`../artifacts/s2c/runs/protocol_v2_textoir_v1/adaptive_v1/contract_repair5/`
- 运行完整性：`../artifacts/s2c/runs/protocol_v2_textoir_v1/adaptive_v1/contract_repair5/ADAPTIVE_V1_VERIFY.json`
- 结果 manifest：`../artifacts/s2c/runs/protocol_v2_textoir_v1/adaptive_v1/contract_repair5/ADAPTIVE_V1_RESULT_MANIFEST.json`
- 诊断：`../artifacts/s2c/runs/protocol_v2_textoir_v1/adaptive_v1/contract_repair5/diagnostics/`
- 轻量结果：`results/diagnostics/adaptive_v1/`
- 主结果表：`results/diagnostics/adaptive_v1/main_results.csv`
- 复现实验命令：`docs/adaptive_v1/REPRODUCE_ADAPTIVE_V1.md`
