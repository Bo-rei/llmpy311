# 可复现性

`protocol_v2_textoir_v1` 是唯一活动实验版本。其 source/canonical/registry/view/export/run 都使用
同一个 versioned 路径；历史协议只能通过显式 `S2C_DATASET_VERSION` 读取，不能与当前结果混用。

## E0：数据和运行时独立性

```bash
cd /home/bo/bo01/llmpy311/s2c
python -m protocol_v2.data.import_textoir --dataset clinc150 --dataset banking77 --dataset stackoverflow
python -m protocol_v2.data.build_canonical
python -m protocol_v2.data.build_registries --seed 13 --seed 42 --seed 87 --seed 100 --seed 123
python -m protocol_v2.data.build_views --seed 13 --seed 42 --seed 87 --seed 100 --seed 123
python -m protocol_v2.data.export_protocol --seed 13 --seed 42 --seed 87 --seed 100 --seed 123
python -m protocol_v2.data.validate_protocol --require-views --require-exports
```

导入完成后临时隐藏 `../textoir` 仍必须能完成 validate、Gate dry-run 和 Gate view loading；这证明
TEXTOIR 只在 import 阶段被读取。

## E1 与 E2

```bash
python -m protocol_v2.experiments.runner \
  --config configs/experiments/protocol_v2_textoir_v1/smoke_gate.yaml --resume
python -m protocol_v2.experiments.verify \
  --config configs/experiments/protocol_v2_textoir_v1/smoke_gate.yaml --require-complete
python -m protocol_v2.experiments.runner \
  --config configs/experiments/protocol_v2_textoir_v1/gate_core_dense.yaml --resume
```

E1 是 36 个 Gate smoke；E2 是 1,650 个可恢复的 frozen-MiniLM Gate 单元。E3--E7 不得在 E2
汇总前启动。每个正式单元写入 config、输入 hash、registry SHA、encoder fingerprint、metrics 和
manifest；没有这些证据的数字不得进入论文表。
