# Historical Chain Reproduction Bundle Design

> Goal: recover and freeze the historical-best CLINC150@KIR50 experiment into a replay-validated research reproduction bundle that can reproduce the previously achieved result or precisely identify the minimal blocking drift.

## 1. Problem Statement

The repository currently mixes at least two different experiment protocols:

- the historical-best chain tied to `data/v19`
- the later generalized multi-dataset chain tied to `data/multidataset/v19/...`

This has created a false appearance that "the historical experiment has been organized" while the actual execution path has drifted in several critical places, especially inside the Gate family.

The immediate task is therefore not code cleanup in the abstract. The task is to reconstruct the exact historical chain as a research-grade reproduction bundle, rerun it under a fresh output directory, and mark each involved file by evidential status.

## 2. Non-Negotiable Objective

The bundle must optimize for exact historical reproduction first.

Primary target:

- reproduce the historical-best CLINC150@KIR50 result represented by:
  - `overall_accuracy = 0.8678`
  - `macro_f1 = 0.8121`
  - `known_intent_accuracy = 0.7995`
  - `oos_f1 = 0.9096`
  - `gate_oos_rejection = 0.9151`
  - `gate_id_recall = 0.8599`

Secondary target:

- if exact replay is blocked, identify the smallest source of drift with hard evidence:
  - data protocol drift
  - gate encoder drift
  - gate training logic drift
  - prototype payload drift
  - checkpoint loading drift
  - evaluation config drift

This work is considered incomplete if it only produces documentation but does not attempt a full replay.

## 3. Historical System Definition

Top-level architecture remains exactly three stages:

1. Gate
2. Router
3. Expert

`prototype` and `semantic verifier` are not standalone stages. They are Gate-family optimizations and must be documented under the Gate family only.

The historical-best chain is defined as the archived CLINC150@KIR50 protocol rooted in the following truth anchors:

- data root: `data/v19`
- known-intent split: `data/v19/KNOWN_INTENTS.json`
- historical reference result:
  `outputs/experiments/archive/sweeps/2026-03-23/pipeline_phase3_proto_eval/pipeline_v19_phase3_proto_d_narrow_t085_eval/eval_results.json`
- frozen reproduction result:
  `outputs/experiments/pipeline/frozen_prototype_gate/prototype_gate_frozen_2026-04-09/prototype_gate_pipeline_frozen/eval_results.json`

## 4. Core Design Principle

Only files with direct evidential value to the historical-best chain belong to the reproduction bundle.

The bundle does not try to reorganize the whole repository. It extracts and validates the historical mother chain.

All files involved in the historical chain must be classified into one of four statuses:

- `VERIFIED_MAINCHAIN`
- `FROZEN_DEPENDENCY`
- `REFERENCE_ONLY`
- `DRIFTED_OR_UNCERTAIN`

## 5. Bundle Structure

The reproduction bundle will be organized conceptually as:

```text
historical_repro_bundle/
├── 00_truth_anchors/
├── 01_data_protocol/
├── 02_gate_family/
├── 03_router/
├── 04_expert/
├── 05_pipeline_eval/
├── 06_replay_validation/
├── 07_reference_only/
└── INDEX.md
```

This is a documentation and validation structure first. It does not require physically moving production code before replay confidence exists.

## 6. File Layers

### Tier 0: Truth Anchors

- historical and frozen `eval_results.json`
- `data/v19/KNOWN_INTENTS.json`
- `data/v19/gate/*.json`
- `data/v19/router/*.json`
- `data/v19/experts/*/*.json`
- `configs/v19/clinc150_historical_best_reference.json`

### Tier 1: Mainchain Entrypoints

- `scripts/data/active/rebuild_multi_dataset_v19.py`
- `tools/gate/train_multisphere_corrected.py`
- `tools/train/train_router_v19.py`
- `tools/train/train_all_experts_v19.py`
- `tools/train/train_expert_v19.py`
- `tools/train/train_semantic_verifier_v19.py`
- `src/pipeline/system_pipeline.py`
- `tools/eval/eval_system_pipeline_v19.py`
- `tools/analysis/run_prototype_gate_frozen_baseline_v19.py`

### Tier 2: Mainchain Dependencies

- `src/gate/multi_sphere_oos_detector.py`
- `src/gate/multi_prototype_gate.py`
- `src/gate/intent_prototype_matcher.py`
- `src/gate/llm_semantic_verifier.py`
- `src/models/architecture.py`
- `src/router/router_model.py`
- `tools/analysis/prototype_path_utils.py`
- `tools/analysis/component_path_utils.py`

### Tier 3: Evidence and Mapping

- `tools/analysis/validate_historical_best_chain_v19.py`
- `tools/analysis/historical_best_pipeline_v19.py`
- `tools/analysis/s2c_code_inventory_v19.py`
- `docs/prototype_gate_historical_best_code_map.md`
- `outputs/reports/historical_best_validation/*`
- `archive/reorg_2026-04-13/repro/historical_best_clinc150_kir50_py/*`

### Tier 4: Reference Only

- multi-dataset orchestration code
- generalized retraining artifacts
- archive diagnostics not exercised by replay

## 7. Replay Strategy

Replay is executed in strict historical protocol mode.

Fixed protocol:

- `data_root = data/v19`
- `KNOWN_INTENTS = data/v19/KNOWN_INTENTS.json`
- Gate encoder must remain historical; current evidence indicates `all-MiniLM-L6-v2`
- new outputs must go to a fresh validation directory
- old artifacts are retained as read-only evidence

Replay sequence:

1. freeze truth anchors
2. validate historical data protocol
3. replay Gate family
4. replay Router
5. replay Expert
6. replay pipeline assembly
7. replay end-to-end evaluation
8. run coverage audit on files actually exercised

## 8. Recovery Ladder

Because the user previously achieved the historical result, failure to reproduce now is treated as repository drift, not as a reason to weaken the goal.

Therefore replay must use this recovery ladder:

1. attempt replay with current active code under historical protocol
2. if metrics drift, compare against archived truth anchors and frozen artifacts
3. identify the first diverging stage
4. if a stage has clearly drifted, restore the historical behavior by preferring archived or legacy-equivalent code for the historical bundle only
5. rerun from that stage forward

This means the reproduction bundle is allowed to depend on archived historical code when active code is no longer behaviorally identical.

## 9. Acceptance Criteria

### Level 1: Inventory Acceptance

- all historically relevant files are identified and classified
- the three-stage architecture is documented correctly
- Gate-family optimizations are documented under Gate, not as standalone stages

### Level 2: Replay Acceptance

- a fresh replay directory is created
- the historical chain is executed end-to-end as far as current code allows
- all generated artifacts are kept separate from legacy outputs

### Level 3: Reproduction Acceptance

One of the following must be achieved:

- exact or near-exact reproduction of the historical-best result using replayed artifacts
- or a precise, stage-localized explanation for why exact reproduction is currently blocked

### Level 4: Research Bundle Acceptance

The final bundle must let the user answer all of the following:

- which Python files and artifacts produced the historical-best result
- which of those files are still trustworthy today
- which stages can be replayed from source
- which stages require frozen or archived dependencies
- where current active code has drifted from historical behavior

## 10. Explicit Anti-Goals

This project does not currently optimize for:

- generalized multi-dataset cleanup
- unified production-style entrypoints
- broad repository refactoring
- deleting historical artifacts

Those may happen later, but only after the historical-best chain is recovered and frozen.
