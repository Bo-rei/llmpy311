# Prototype Gate 历史最优复现代码总览（CLINC150 + KIR50）

本文档面向复现目标：

- 数据集：CLINC150
- KIR：0.50
- 历史最优指标：
  - overall_acc: 0.8678
  - macro_f1: 0.8121
  - known_acc: 0.7995
  - oos_f1: 0.9096
  - gate_oos_rej: 0.9151
  - gate_id_recall: 0.8599

## 1. 历史最优结果文件（基准真值）

- outputs/experiments/archive/sweeps/2026-03-23/pipeline_phase3_proto_eval/pipeline_v19_phase3_proto_d_narrow_t085_eval/eval_results.json
- outputs/experiments/pipeline/regression/regression_guardrail_20260323/phase3_proto_narrow_eval/eval_results.json
- outputs/experiments/pipeline/evals/pipeline_v19_stage2_proto_ref_eval/eval_results.json
- outputs/experiments/pipeline/frozen_prototype_gate/prototype_gate_frozen_2026-04-09/prototype_gate_pipeline_frozen/eval_results.json

## 2. 数据处理相关文件

### 2.0 严格历史协议与推广协议的边界

当前仓库内存在两套不同口径：

- 严格历史复现链：`data/v19`
- 后续推广/多数据集链：`data/multidataset/v19/...`

本文件服务于“历史最优结果复现”，因此默认优先级如下：

1. `data/v19`
2. 历史冻结工件
3. 当前 active 代码
4. 推广链与多数据集脚本

如果某个 active 脚本已经和历史行为漂移，则它只能作为参考，不应直接等同为历史最优主链。

### 2.1 多数据集/KIR切分与重建

- scripts/data/active/rebuild_multi_dataset_v19.py
- scripts/data/active/build_known_intents_ratio_splits_v19.py
- scripts/data/active/rebuild_v19_2_strict.py
- scripts/data/active/audit_v19_2_data.py

### 2.2 数据产物（严格历史协议）

- data/v19/KNOWN_INTENTS.json
- data/v19/gate/train.json
- data/v19/gate/val.json
- data/v19/gate/test.json
- data/v19/router/train.json
- data/v19/router/test.json
- data/v19/experts/*/train.json
- data/v19/experts/*/val.json
- data/v19/experts/*/test.json

### 2.3 数据产物（推广链参考，不作为历史真值）

- data/multidataset/v19/index.json
- data/multidataset/v19/clinc150/kir50_seed42/MANIFEST.json
- data/multidataset/v19/clinc150/kir50_seed42/AUDIT.json
- data/multidataset/v19/clinc150/kir50_seed42/gate/train.json
- data/multidataset/v19/clinc150/kir50_seed42/gate/val.json
- data/multidataset/v19/clinc150/kir50_seed42/gate/test.json
- data/multidataset/v19/clinc150/kir50_seed42/router/train.json
- data/multidataset/v19/clinc150/kir50_seed42/router/val.json
- data/multidataset/v19/clinc150/kir50_seed42/router/test.json
- data/multidataset/v19/clinc150/kir50_seed42/experts/*/train.json
- data/multidataset/v19/clinc150/kir50_seed42/experts/*/val.json
- data/multidataset/v19/clinc150/kir50_seed42/experts/*/test.json

## 3. 模型训练相关文件

### 3.1 Gate 训练

- tools/gate/train_multisphere_corrected.py
- tools/archive/code_cleanup_2026-03-28/legacy_tools/gate/train_multisphere_corrected.py

### 3.2 Prototype 构建（历史最优链路核心）

- tools/archive/code_cleanup_2026-03-28/legacy_tools/gate/build_multi_prototypes_v19.py
- tools/archive/code_cleanup_2026-03-28/legacy_tools/gate/build_intent_prototypes_v19.py
- tools/archive/code_cleanup_2026-03-28/legacy_tools/gate/expand_selective_prototypes_v19.py
- tools/analysis/historical_best_pipeline_v19.py
- tools/analysis/prototype_path_utils.py

### 3.3 Router 训练

- tools/train/train_router_v19.py
- tools/archive/code_cleanup_2026-03-28/legacy_tools/train/train_router_v19.py

### 3.4 Expert 训练

- tools/train/train_expert_v19.py
- tools/train/train_all_experts_v19.py
- tools/archive/code_cleanup_2026-03-28/legacy_tools/train/train_expert_v19.py
- tools/archive/code_cleanup_2026-03-28/legacy_tools/train/train_all_experts_v19.py

### 3.5 Gate 后置语义判别（属于 gate 子模块）

- tools/train/train_semantic_verifier_v19.py
- src/gate/llm_semantic_verifier.py
- src/gate/intent_prototype_matcher.py

## 4. Pipeline 组装相关文件

### 4.1 核心推理流水线

- src/pipeline/system_pipeline.py
- src/router/router_model.py
- src/models/architecture.py
- src/gate/multi_prototype_gate.py
- src/gate/multi_sphere_oos_detector.py

### 4.2 历史最优复现入口（推荐）

- tools/analysis/run_prototype_gate_frozen_baseline_v19.py
- tools/analysis/validate_historical_best_chain_v19.py
- archive/reorg_2026-04-13/repro/historical_best_clinc150_kir50_py/repro_entry.py
- tools/analysis/component_path_utils.py
- tools/analysis/prototype_path_utils.py

## 5. 评估相关文件

### 5.1 系统评估入口

- tools/eval/eval_system_pipeline_v19.py
- tools/analysis/run_multi_dataset_benchmark_v19.py

### 5.2 诊断与辅助评估

- tools/archive/code_cleanup_2026-03-28/eval/eval_gate_test.py
- tools/archive/code_cleanup_2026-03-28/eval/eval_gate_lightweight.py
- tools/archive/code_cleanup_2026-03-28/eval/evaluate_gate_quantile.py
- tools/archive/code_cleanup_2026-03-28/eval/diagnose_system_v19.py

## 6. 编排与实验管理

- tools/analysis/run_multi_dataset_training_v19.py
- outputs/experiments/multi_dataset_v19/<dataset_slug>/<kirXX_seed42>/train_manifest.json
- outputs/experiments/multi_dataset_v19/<dataset_slug>/<kirXX_seed42>/<stage>/stage_manifest.json
- outputs/results_summary/aggregate_results.py
- outputs/results_summary/master_results.json

目录命名说明：
- `dataset_slug` 使用小写目录：`clinc150`、`banking77_oos`、`snips`。
- 历史大写目录（如 `CLINC150_kir50_seed42`）仅作为兼容遗留，不再作为主线输出口径。

## 7. 与历史最优复现直接关联的最小闭环

如果只看“复现历史最优 Prototype Gate（CLINC150@KIR50）”的最小链路，重点文件如下：

1. 数据重建：scripts/data/active/rebuild_multi_dataset_v19.py
2. Gate训练：tools/archive/code_cleanup_2026-03-28/legacy_tools/gate/train_multisphere_corrected.py
3. Router训练：tools/archive/code_cleanup_2026-03-28/legacy_tools/train/train_router_v19.py
4. Expert训练：tools/archive/code_cleanup_2026-03-28/legacy_tools/train/train_all_experts_v19.py
5. Prototype路径解析：tools/analysis/prototype_path_utils.py
6. 冻结复现入口：tools/analysis/run_prototype_gate_frozen_baseline_v19.py
7. 系统评估：tools/eval/eval_system_pipeline_v19.py
8. 对标真值：outputs/experiments/archive/sweeps/2026-03-23/pipeline_phase3_proto_eval/pipeline_v19_phase3_proto_d_narrow_t085_eval/eval_results.json
9. 母链验证与文档：tools/analysis/validate_historical_best_chain_v19.py
10. 规范化 reference：configs/v19/clinc150_historical_best_reference.json

## 7.1 机器可读母链报告

建议优先查看下面两份新产物，它们把历史最优结果对应的代码、数据、模型和参数链整理成统一视图：

- `outputs/reports/historical_best_validation/historical_best_mother_chain.json`
- `outputs/reports/historical_best_validation/historical_best_mother_chain.md`

## 8. 备注

- 业务流程口径为三阶段：gate -> router -> expert。
- train_semantic_verifier_v19.py 是 gate 的后置语义判别子模块，不作为并列主阶段。
- 当前 replay 验证应优先使用 `tools/analysis/replay_historical_chain_v19.py` 与 `docs/historical_repro_bundle/INDEX.md`，它们会明确区分 `VERIFIED_MAINCHAIN`、`FROZEN_DEPENDENCY`、`REFERENCE_ONLY` 和 `DRIFTED_OR_UNCERTAIN`。
- 复现时务必保持 Router LoRA 配置与后置语义判别加载 Router 的 LoRA 配置一致，避免 checkpoint 维度不匹配。
- 9 组多数据集 / 多 KIR 实验应统一引用 `tools/analysis/historical_best_pipeline_v19.py` 中的历史最优 profile；可变项仅保留数据集与 KIR，其他方法参数不再分叉。
