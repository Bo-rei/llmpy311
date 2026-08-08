# 训练参与式自适应多中心：合同修复实验报告

更新时间：2026-08-05  
活动协议：`protocol_v2_textoir_v1`  
阶段：`joint_adaptive_multicenter_contract_repair_v1`  
有效 attempt：`repair3`

## 结论先行

本阶段真正训练了候选第二中心：每个 seed 都从 RACAL Trainable K=1 checkpoint 出发，联合更新
MiniLM 最后两层、残差 projection 和 intent prototypes；候选结构只由 Known train/calibration
决定。三个 seed 的候选都被安全门拒绝，最终每个 Known intent 都回退到 `K_y=1`。因此，当前
StackOverflow 条件下没有得到可接受的训练参与式多中心增益，但已经排除了“代码没有训练第二中心”
这一解释。

## 1. 修复了什么

旧 pilot 的候选比较存在三项需要隔离的契约风险，本阶段单独修复：

1. K=1 父边界在候选产生前拟合一次，候选阶段冻结，不因候选表示变化而重估父边界；
2. compactness 使用 parent-guarded score，不使用会绕过父边界的 unconstrained score；
3. 候选训练增加子中心负载平衡和中心分离项，并记录子簇负载、最小分离和半径。

所有选择均只使用 Known train/calibration；`test_used_for_selection=false`、
`oos_used_for_training=false`。

## 2. 实验设置

| 项目 | 设置 |
|---|---|
| 数据集 | StackOverflow |
| KIR | 0.50 |
| seeds | 13、42、87 |
| 初始模型 | RACAL Trainable K=1 checkpoint |
| 候选结构 | 每轮最多一个 PCA 残差二分 |
| 候选训练 | 2 epochs，encoder + projection + prototypes |
| 父边界 | K=1 固定父边界 |
| 选择数据 | Known train + Known calibration |
| 测试 OOS | 只用于最终评价 |
| 完成情况 | 3/3，失败 0 |

## 3. 结构选择结果

| seed | 候选 intent | 子簇负载 | 最小中心分离 | Calibration Known Recall 前→后 | 决策 |
|---:|---|---|---:|---:|---|
| 13 | osx | 452 / 148 | 0.0325 | 0.826 → 0.770 | reject：Recall drop |
| 42 | cocoa | 138 / 462 | 0.0576 | 0.819 → 0.724 | reject：Recall drop |
| 87 | osx | 276 / 324 | 0.0376 | 0.850 → 0.722 | reject：Recall drop |

候选并非空跑：每个 seed 都产生了候选 checkpoint 和训练 history；失败发生在 Known-only
结构安全门，而不是训练入口未执行。

## 4. 最终 Gate 结果

| 方法 | OOS F1 | F1-All | F1-K | Accuracy | Known Recall | False Acceptance | AUROC | AUPR-OOS | 平均 K_y |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Contract-repair adaptive | 0.8661 ± 0.0091 | 0.8563 ± 0.0041 | 0.8553 ± 0.0038 | 0.8577 ± 0.0073 | 0.8388 ± 0.0037 | 0.1129 ± 0.0187 | 0.8977 ± 0.0024 | 0.8396 ± 0.0021 | 1.0 |

它与 RACAL Trainable K=1 的结果几乎相同，是因为三个候选都被拒绝，最终结构严格回退到 K=1。

## 5. 研究解释

- 可以确认：训练参与式的候选多中心链路真实存在并运行；
- 可以确认：在当前 StackOverflow/KIR=.50 的表示与边界合同下，候选二分造成明显 Known calibration
  Recall 损失；
- 不能确认：所有训练参与式自适应多中心都失败；本阶段只测试一个候选分裂规则、一个数据集和一个 KIR；
- 不能宣称：当前方法已经是成功的 adaptive-K 或 SOTA；
- 当前最强自有结论仍是 Known-only Trainable MiniLM + K=1 Gate，而不是多中心。

## 6. 证据路径

- 运行根：`../artifacts/s2c/runs/protocol_v2_textoir_v1/joint_adaptive_multicenter_contract_repair_v1/repair3/`
- Provenance：`JOINT_ADAPTIVE_CONTRACT_REPAIR_PROVENANCE.json`
- 汇总：`CONTRACT_REPAIR_SUMMARY.json`
- 完整性：`run_joint_adaptive_contract_repair_v1.py verify`
- 配置：`configs/experiments/protocol_v2_textoir_v1/joint_adaptive_multicenter_contract_repair_v1.yaml`

## 7. 决策

停止本阶段，不扩展 KIR、数据集、K=3--5、Proxy-OOS 或完整 Pipeline。若继续研究训练参与式
自适应多中心，必须先登记新的目标函数/分裂规则，并在同一规模 pilot 中验证；否则保留本结果作为
StackOverflow 上多中心安全门失败的机制证据。
