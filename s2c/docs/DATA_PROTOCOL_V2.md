# protocol_v2_textoir_v1 数据契约

活动协议将输入快照、Known-intent 抽样、固定 views、方法导出和实验结果按版本隔离。
它不修改 TEXTOIR 文本、标签或 train/dev/test split，也不对 StackOverflow 去重。

## 物化链

```text
textoir/data (仅导入)
  -> data/sources/textoir/<commit>/
  -> data/canonical/protocol_v2_textoir_v1/
  -> data/registries/protocol_v2_textoir_v1/
  -> data/views/protocol_v2_textoir_v1/
  -> data/exports/protocol_v2_textoir_v1/
  -> artifacts/s2c/runs/protocol_v2_textoir_v1/
```

所有复制逐文件 SHA256 一致；canonical 记录保留原始 text、intent、split、source row 和确定性
`sample_id`。连续构建必须产生相同 canonical SHA256。

## KIR registry

正式 KIR 是 `.10,.20,.25,.30,.40,.50,.60,.70,.75,.80,.90`，正式 seed 是
`13,42,87,100,123`。Known intents 使用 TEXTOIR 的 `benchmark_labels` 固定顺序，并按
`numpy.random.seed(seed)` 加无放回 `numpy.random.choice` 抽样。所有方法只能读取 registry，
不得重新抽样。

## Views 与模型选择

- `train_known`：仅 Known train。
- `calibration_known`：仅 Known dev。
- `test_known`、`test_heldout_oos`、`test_native_oos`、`test_combined`：只用于最终评估。

K、半径、边界与阈值不得使用 held-out/native OOS dev 或任何 test OOS。CLINC 的 native OOS
与 held-out intent OOS 始终分开保留；Banking77/StackOverflow 的 native OOS 数为 0。

## 统一导出

s2c、TEXTOIR TSV、MOGB TSV、K+1-way、ADB TSV 和 DA-ADB TSV 都从同一 canonical+registry+views
生成。ADB/DA-ADB 的 export 只是输入契约，不等价于已复现这些方法。

运行前必须通过：

```bash
python -m s2c.data.validate_protocol --require-views --require-exports
python tools/maintenance/check_data_tracking.py
```
