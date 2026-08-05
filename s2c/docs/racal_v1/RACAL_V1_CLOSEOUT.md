# RACAL-v1 阶段一收口

## 阶段

- 活动协议：`protocol_v2_textoir_v1`
- 阶段：`racal_v1_stage1`
- 数据集：StackOverflow
- KIR：0.50
- 数据种子：13、42、87
- 允许的方法：Frozen K=1 精确回放、Trainable MiniLM K=1

## 完成情况

- Frozen K=1 E2 精确回放：3/3。
- Trainable MiniLM K=1：3/3。
- 总运行单元：6/6，失败、缺失、重复和无效指标均为 0。
- 未运行固定 K=2、中心激活、Proxy-OOS、其他数据集/KIR、Gate–Router–Expert 或外部基线。

## 关键证据

Frozen 与 Trainable 的三 seed 均值如下：

| 方法 | OOS F1 | F1-All | F1-K | Accuracy | Known Recall | False acceptance | AUROC | AUPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen K=1 | 0.77290 | 0.78598 | 0.78728 | 0.75300 | 0.83711 | 0.26544 | 0.88191 | 0.84684 |
| Trainable K=1 | 0.86709 | 0.85654 | 0.85548 | 0.85822 | 0.83922 | 0.11144 | 0.91751 | 0.88777 |
| Trainable - Frozen | +0.09419 | +0.07056 | +0.06820 | +0.10522 | +0.00211 | -0.15400 | +0.03560 | +0.04093 |

所有训练 checkpoint 都使用 Known calibration 选择；测试 OOS 只在模型和边界冻结后评价。

## 阶段决策

RACAL-v1 阶段一的 K=1 表示适配通过预注册晋级门：OOS F1 和 F1-All 提升，Known Recall 变化不超过门槛，且三 seed 方向一致。因此可以登记下一阶段“Trainable representation + 风险校准多中心”的计划。

本收口不授权自动运行下一阶段。下一阶段必须另行固定协议、记录新的 provenance，并首先用同一 StackOverflow/KIR=0.50/三 seed 设计一个受控 K=2 对照；若 K=2 再次出现 false acceptance 爆炸，应停止固定多中心扩展，而不是继续枚举 K 或调损失。

## 复现入口

- 运行说明：`docs/racal_v1/REPRODUCE_RACAL_V1.md`
- 阶段报告：`docs/racal_v1/RACAL_V1_REPORT.md`
- 轻量结果：`results/diagnostics/racal_v1/RACAL_V1_STAGE1_MEAN_STD.csv`
- 运行产物：`../artifacts/s2c/runs/protocol_v2_textoir_v1/racal_v1/`
- 运行验证：`../artifacts/s2c/runs/protocol_v2_textoir_v1/racal_v1/RACAL_VERIFY.json`

## 已知环境状态

`third_party/mogb_official` 的 dirty 状态是进入本阶段前已存在的只读审计元数据和缓存；RACAL-v1 没有修改该目录。完整 Git 状态和这一例外必须在最终实验记录中保留。
