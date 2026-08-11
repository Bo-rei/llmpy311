# DCLOOS 当前协议单格阻塞记录 V1

更新时间：2026-08-10
协议：`protocol_v2_textoir_v1`
工作点：StackOverflow / KIR=0.50 / seed=42

## 状态

本次运行已经使用当前 StackOverflow 的 train/dev/test 快照和 registry 的固定 Known 列表启动了
DCLOOS BERT 训练，并保留了 DCLOOS 的 pseudo-OOS 与外部 SQuAD 监督。运行约 3530 秒后进程结束，
没有生成最终 `metrics.json`、`stdout.log` 或 `stderr.log`。现有 `predictions.npz` 只是中间 validation-best
预测，不能作为性能结果。

## 证据

- 运行根：`../artifacts/s2c/external/dcloos_stackoverflow_kir050_seed42_fixed_registry_v1/`
- 状态 manifest：同一目录的 `run_manifest.json`
- 中间预测：同一目录的 `predictions.npz`
- 外部 SQuAD SHA256：`f6bf61866c86d3b11565826c3ca1faa00e31f196e0ad9bfd000ec45575fd426e`
- 已确认 GPU 训练曾运行，显存约 11.8GB；该问题不是“没有启动训练”。

## 不能得出的结论

1. 不能用中间预测计算 DCLOOS OOS F1。
2. 不能把既有 KIR=.75/seed=888 reduced 结果改写成当前 protocol_v2 结果。
3. 不能据此判断 DCLOOS 优于或劣于 S2C、MOGB、ADB 或 DA-ADB。

## 当前处理

这次单格记为 `interrupted_no_final_metrics`，保留 artifact，不重跑更大矩阵。DCLOOS 仍作为
额外未知监督的外部参照；当前可解释的主证据继续使用已完成的 S2C fair 矩阵、MOGB 组件分析、
ADB 45/45 和 DA-ADB 当前三 seed 结果。

## 后续 reduced 预算尝试（同样未形成性能结果）

另启动 `dcloos_stackoverflow_kir050_seed42_fixed_registry_reduced10_v1`：固定同一 registry Known 列表、
`max_epochs=10`、`patient=3`。该运行在约 1,246 秒后仍未完成，实际走 CPU 路径，随后人工停止；
runner 已写入 `run_manifest.json`，状态为 `timeout_incomplete`。`predictions.npz` 仍是中间
validation-best prediction，继续排除在所有指标和图表之外。该尝试只增加资源诊断证据，不改变
DCLOOS 的算法结论，也不替代既有 reduced 兼容参考。

## 2026-08-10：固定 Registry GPU 预算尝试收口

- 新运行根：`../artifacts/s2c/external/dcloos_stackoverflow_kir050_seed42_fixed_registry_gpu_v1/`。
- 运行配置：固定 protocol_v2 Known 列表、BERT、本地 SQuAD 外部负样本、`max_epochs=10`、`patient=3`、GPU 0；外部负样本 SHA256 仍为
  `f6bf61866c86d3b11565826c3ca1faa00e31f196e0ad9bfd000ec45575fd426e`。
- 资源证据：训练进程实际使用 RTX 5070，峰值显存约 11.8GB；因此本次问题不是 CUDA 未启用。
- 结果：在预先声明的 30 分钟运行上限内人工停止；manifest 状态为 `timeout_incomplete`，没有生成最终 `metrics.json`。
- 中间文件：`predictions.npz` 仅为 validation-best 中间预测，manifest 明确标记 `intermediate_predictions_excluded=true`，不得进入任何汇总、图表或性能结论。
- provenance：`run_manifest.json` SHA256=`7350346fa602eb8bbb1e7d36caff4b1bfe0ed262422821913b8619f81e725d9f`；中间预测 SHA256=`fa4121110028cf1604b07f3a02ac47a429d9c15ff5eb7d4d923e27a5f40c88a9`。

本次尝试只增加了“固定 Registry 的 DCLOOS 在 GPU 上仍未在当前预算内完成”的资源诊断，不能证明 DCLOOS 性能，也不改变既有 reduced KIR=.75/seed=888 参考的合同边界。后续若要继续，必须先改为可流式/可恢复的独立预算方案并重新登记实验；不得重复同一命令或把中间预测当作结果。
