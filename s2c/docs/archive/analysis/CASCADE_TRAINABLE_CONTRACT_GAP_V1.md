# Trainable Gate 与旧 Cascade 的数据合同断点（V1）

更新时间：2026-08-10  
分析阶段：`cascade_trainable_contract_gap_v1`  
目的：阻止将 `protocol_v2_textoir_v1` 的 Trainable Gate 与旧 `v19` Router/Expert 直接混合。

## 结论

当前不能直接生成“Trainable MiniLM 完整 Cascade”结果。原因不是 Trainable checkpoint 缺失，而是
现有完整 Cascade 的下游组件和数据不是同一协议：Trainable checkpoint 使用 TEXTOIR canonical/views，
而 Router/Expert 来自 v19 prepared data。若直接替换 Gate encoder，会同时改变输入样本、Known 列表、
split 和下游训练分布，结果无法归因。

## 证据

| 项目 | protocol_v2 Trainable | v19 Cascade components |
|---|---:|---:|
| train 行数 | 6000 | 5995 |
| calibration/val 行数 | 1000 | 1998 |
| test 行数 | 6000 | 5990 |
| test Known 行数 | 3000 | 2993 |
| test OOS 行数 | 3000 | 2997 |
| 数据协议 | `protocol_v2_textoir_v1` | `v19_prepared_data` |
| Trainable test sample hash | `ed50cc34f4a8e69711527e7db4d5bed877b2e768861d442975d1a7b4907b16c6` | `无 sample_id 字段` |
| v19 test source_id hash | — | `df200181cea24298550602dec32c79b6e1f614b75025e138427acb3641225ee8` |

v19 的 gate JSON 只有 `text/intent/domain/split/label/source_id/title/question/tag` 等字段，没有当前
protocol 的 `sample_id`。即使文本数量接近，也不能视为同一测试集合。

下游组件清单：

- component plan：`cascade_full_kir50_downstream_components`
- source：`existing_seed42_audited`
- Router exists：`True`
- Expert root exists：`True`

## 对当前结果的影响

因此当前结果必须分为：

1. Trainable K=1：当前协议的 Gate-only 结果；
2. Frozen/CE-Recon Cascade：旧 v19 下游组件的 Cascade 结果；
3. 历史 fulltex Ours：历史旧合同的完整 Cascade；
4. 外部 MOGB/ADB/DCLOOS：各自独立监督和数据合同。

不能把第 1 项直接和第 2 项合成同协议端到端排名。

## 下一项真正需要运行的实验

若要回答“Trainable 表示是否改善完整 Cascade”，必须先在当前 `protocol_v2_textoir_v1` 的同一
registry/views 上重新训练 Router/Expert，然后固定这些下游 checkpoint，比较：

```text
Frozen K=1 Cascade
Trainable K=1 Cascade
```

首轮建议 StackOverflow/KIR=.50/seeds=(13, 42, 87)，之后再扩展其他数据集。训练和评价期间不得读取
旧 v19 test 文件，也不能复用 v19 Router/Expert 冒充同协议结果。

## 产物

- `results/analysis/archive/analysis/cascade_trainable_contract_gap_v1/contract_comparison.csv`
- `results/analysis/archive/analysis/cascade_trainable_contract_gap_v1/component_manifest.json`
- `figures/archive/analysis/cascade_trainable_contract_gap_v1/split_count_contract_gap.png`

这是合同审计，不是模型失败结论；在下游组件迁移完成前，Trainable 的 Gate-only 结果仍然是当前
最可靠的自有候选。
