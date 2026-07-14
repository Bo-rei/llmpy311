# Historical Chain Total Ledger

## 1. Scope

This ledger is for the historical CLINC150@KIR50 chain only.

Top-level architecture stays:

1. Gate
2. Router
3. Expert

`prototype` and `semantic verifier` are Gate-family optimizations, not separate top-level stages.

More specifically:

- `prototype` is a geometric Gate refinement over intent-level embedding centers.
- `semantic verifier` is a Gate-internal decoder-only LLM fallback for uncertain cases.
- Neither should be documented as a fourth standalone component beside Gate, Router, and Expert.

## 2. What Was Actually Verified

The replay completed under strict historical protocol:

- `data_root = data/v19`
- `known_intents_path = data/v19/KNOWN_INTENTS.json`
- `gate_encoder_path = all-MiniLM-L6-v2`
- `device = cuda`

Replay artifacts:

- `outputs/reports/historical_replay_20260414/replay_manifest.json`
- `outputs/reports/historical_replay_20260414/frozen_eval/eval_results.json`
- `outputs/reports/historical_replay_20260414/frozen_eval/predictions.json`

Important boundary:

- This replay was **not** a full retraining run.
- It was a **frozen historical evaluation replay** using archived detector/router/experts artifacts.
- Therefore it proves the historical evaluation chain can be recovered, but it does **not** by itself prove retraining is already clean.

## 3. Historical Truth Anchors

- Reference eval:
  `outputs/experiments/archive/sweeps/2026-03-23/pipeline_phase3_proto_eval/pipeline_v19_phase3_proto_d_narrow_t085_eval/eval_results.json`
- Frozen historical detector:
  `outputs/experiments/archive/sweeps/2026-03-23/gate_l2_mix2_train/gate_l2_mix2_true_lambda_1p6/detector.json`
- Historical known intents:
  `data/v19/KNOWN_INTENTS.json`

## 4. Mainchain Code Ledger

### Data Protocol

- `scripts/data/active/rebuild_multi_dataset_v19.py`
- `data/v19/KNOWN_INTENTS.json`
- `data/v19/gate/*.json`
- `data/v19/router/*.json`
- `data/v19/experts/*/*.json`

### Gate Family

- `tools/gate/train_multisphere_corrected.py`
- `tools/train/train_semantic_verifier_v19.py`
- `src/gate/multi_sphere_oos_detector.py`
- `src/gate/multi_prototype_gate.py`
- `src/gate/intent_prototype_matcher.py`
- `src/gate/llm_semantic_verifier.py`

Historical interpretation:

- The first-pass Gate boundary is still built by `all-MiniLM-L6-v2` embeddings plus multisphere / prototype geometry.
- `IntentPrototypeMatcher` is a Gate-side prototype matcher, not a Router or Expert module.
- `LLMSemanticVerifier` exists to use decoder-only LLM reasoning on ambiguous Gate cases where geometric proximity alone is not sufficient to disambiguate intent.

### Router

- `tools/train/train_router_v19.py`
- `src/models/architecture.py`
- `src/router/router_model.py`

### Expert

- `tools/train/train_expert_v19.py`
- `tools/train/train_all_experts_v19.py`
- `src/models/architecture.py`

### Pipeline Assembly and Eval

- `src/pipeline/system_pipeline.py`
- `tools/eval/eval_system_pipeline_v19.py`
- `tools/analysis/run_prototype_gate_frozen_baseline_v19.py`
- `tools/analysis/replay_historical_chain_v19.py`

## 5. Verified Findings From Code Audit

### Finding A: historical replay was evaluation-only, not retraining

Evidence:

- `tools/analysis/replay_historical_chain_v19.py` currently replays the frozen historical evaluation path.
- `outputs/reports/historical_replay_20260414/frozen_eval/eval_results.json` was produced from archived detector/router/experts artifacts.

Impact:

- Historical evaluation chain is validated.
- Historical retraining chain still requires separate verification.

### Finding B: KIR extension was previously broken at Gate training

Evidence:

- `tools/analysis/run_multi_dataset_training_v19.py` accepted `gate_encoder_path` but did not pass it into Gate training.
- The Gate stage used `model_path`, which would incorrectly turn historical Gate training into `smollm135m` embedding training.

Status:

- Fixed.

### Finding C: KIR extension was previously broken at evaluation defaults

Evidence:

- `tools/analysis/run_multi_dataset_benchmark_v19.py` defaulted `gate_encoder_path` to `model_path`.
- That default drifts away from historical Gate-family behavior.

Status:

- Fixed.

### Finding D: Router/Expert training still use `torch.cuda.is_available()`

Evidence:

- `tools/train/train_router_v19.py`
- `tools/train/train_expert_v19.py`

Impact:

- This is a residual environment risk on this machine because the repo guidance says not to rely on `torch.cuda.is_available()` during init.
- It does not block the frozen historical replay, but it is relevant before large KIR retraining runs.

## 6. Current Extension Readiness

Current status before extending to `CLINC150@KIR25/75`:

- Historical frozen replay: verified
- Historical strict protocol constants: verified
- KIR training runner Gate encoder propagation: verified
- KIR benchmark runner Gate encoder default: verified
- Full retraining on KIR25/75: not yet executed

## 7. KIR Diagnostic Findings

The completed `CLINC150@KIR25/75` extension runs show that `KIR25` and `KIR75` are not minor perturbations around the historical `KIR50` protocol.

Observed protocol differences:

- Historical `KIR50` frozen replay operates on a test distribution of roughly `2210 known / 3289 OOS`.
- `KIR25` end-to-end eval operates on roughly `1108 known / 4392 OOS`.
- `KIR75` end-to-end eval operates on roughly `2958 known / 2542 OOS`.

This matters because:

- `overall_accuracy` is not directly comparable across these three KIR settings without considering the changing known/OOS mix.
- `KIR25` appears strong on OOS rejection mainly because the test protocol is much more OOS-heavy.
- The more stable cross-KIR comparisons are `macro_f1`, `known_intent_accuracy`, `gate_oos_rejection`, and `gate_id_recall`.

Observed Gate behavior:

- `KIR25`: over-conservative Gate behavior
  - stronger OOS rejection than historical `KIR50`
  - weaker ID recall than historical `KIR50`
- `KIR75`: weaker balance on both sides
  - lower OOS rejection than historical `KIR50`
  - lower ID recall than historical `KIR50`

Observed semantic fallback behavior:

- The semantic Gate fallback is active on a narrow minority of samples:
  - historical `KIR50`: `509 / 5499`
  - `KIR25`: `544 / 5500`
  - `KIR75`: `569 / 5500`
- Its role remains a second-pass Gate correction layer rather than a primary decision maker.

## 8. Immediate Next Step

Before claiming the KIR25/75 experiments are aligned with the historical method, run:

1. CLINC150 KIR25 training under the fixed Gate encoder path
2. CLINC150 KIR25 benchmark
3. CLINC150 KIR75 training under the fixed Gate encoder path
4. CLINC150 KIR75 benchmark

These runs should remain on the historical method family:

- Gate encoder: `all-MiniLM-L6-v2`
- Router/Expert base model: `smollm135m`
- semantic/prototype settings inherited from `historical_best`
