# DA-ADB 隔离 CUDA 单格审计 V1

## 运行合同

- 数据集：StackOverflow；KIR=`0.50`；seed=`42`。
- 数据根：`/home/bo/bo01/llmpy311/artifacts/s2c/external/adb_protocol_v2_probe_data_v1`；训练/开发/测试 SHA256 记录在 run manifest。
- 表示与训练：BERT `bert_disaware`，TEXTOIR 外部兼容 runner；不是 MiniLM fair 行。
- runtime：`2.9.1+cu128`，Transformers `4.46.3`，RTX 5070 CUDA。
- 测试 OOS 只在最终测试阶段使用；没有用测试 OOS 选 epoch、阈值或参数。

## 审计结果

`run_manifest.json` 为 `complete`，返回码为 0；逐样本预测存在且无 NaN/Inf，真实标签与预测标签形状一致，预测没有塌缩为单一类别。`y_true/y_pred` 的 SHA256、split SHA256 和完整命令均保存在机器可读 manifest 中。

## 指标

| 指标 | 值 |
|---|---:|
| OOS F1 | 70.82% |
| F1-All | 72.03% |
| F1-Known | 72.15% |
| Accuracy | 70.15% |
| Known Recall | 72.93% |
| False Acceptance | 30.33% |
| False Rejection | 27.07% |

## 与既有 DA-ADB 证据的关系

该单格修复了“当前隔离 runtime 只有 NaN/全类预测”的阻断，但它不能自动证明算法性能，也不能覆盖旧 seed=0 的兼容单格。新结果为 OOS F1 `70.82%`，明显低于旧兼容单格的 90.90%；差异必须归因于数据/seed/环境/兼容配置差异并单独审计，不能挑选较高数字。

因此当前 DA-ADB 状态为 `valid_external_cell_pending_replication`：可作为 BERT/TextOIR 外部参照，尚不能与 MiniLM fair matrix 合并排名，也不能据此宣称 SOTA。

机器可读证据：`results/analysis/archive/analysis/da_adb_gpu_runtime_v1/`；原始运行：`/home/bo/bo01/llmpy311/artifacts/s2c/external/da_adb_gpu_runtime_v1/stackoverflow/DA-ADB/kir_0.50/seed_42`。
