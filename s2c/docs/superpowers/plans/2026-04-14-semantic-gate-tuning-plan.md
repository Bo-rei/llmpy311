# Semantic Gate Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add validation-based semantic gate tuning so `llm_verifier` and `fusion` can choose a better Gate operating point before test-time evaluation.

**Architecture:** Keep the verifier model and pipeline structure unchanged. Extend the eval path with a validation-only tuning helper that sweeps semantic threshold for `llm_verifier`, and sweeps threshold plus fusion alpha for `fusion`, then applies the chosen setting to the final test evaluation.

**Tech Stack:** Python 3.11, NumPy, scikit-learn, pytest

---

### Task 1: Add failing tests for semantic gate tuning helpers

**Files:**
- Modify: `tests/historical_repro/test_historical_protocol_v19.py`
- Modify: `tools/eval/eval_system_pipeline_v19.py`

- [ ] **Step 1: Write failing tests**

Add tests that expect:
- a threshold sweep helper to return a better threshold than a bad fixed default on synthetic scores
- the historical profile to expose verifier LoRA settings used by tuning/eval

- [ ] **Step 2: Run the targeted tests to verify failure**

Run: `pytest tests/historical_repro/test_historical_protocol_v19.py -q`
Expected: FAIL because semantic tuning helpers/profile defaults are not fully exposed yet.

### Task 2: Implement validation-only semantic tuning

**Files:**
- Modify: `tools/eval/eval_system_pipeline_v19.py`

- [ ] **Step 1: Add minimal tuning helpers**

Implement helpers to:
- sweep threshold over semantic decision scores and maximize binary macro-F1 on `gate_val`
- for `fusion`, optionally sweep a small alpha grid and threshold grid jointly

- [ ] **Step 2: Wire new CLI flags into eval**

Add:
- `--semantic_tuning_mode {fixed,val_macro_f1}`
- keep `fixed` as default
- if enabled on `llm_verifier` or `fusion`, tune on `gate_val` only and record the selected params in `eval_results.json`

- [ ] **Step 3: Keep prototype historical path unchanged**

Do not change default historical frozen behavior for `semantic_gate_mode=prototype`.

### Task 3: Propagate tuning settings through benchmark orchestration

**Files:**
- Modify: `tools/analysis/historical_best_pipeline_v19.py`
- Modify: `tools/analysis/run_multi_dataset_benchmark_v19.py`

- [ ] **Step 1: Expose semantic tuning defaults in profile**

Add profile defaults for:
- `semantic_tuning_mode`
- verifier LoRA shape already required by eval

- [ ] **Step 2: Forward settings into benchmark eval command**

Ensure benchmark runs can pass semantic tuning flags and verifier LoRA settings through to `tools/eval/eval_system_pipeline_v19.py`.

### Task 4: Verify and measure impact on historical KIR50

**Files:**
- Modify: `tools/eval/eval_system_pipeline_v19.py`
- Modify: `tests/historical_repro/test_historical_protocol_v19.py`

- [ ] **Step 1: Run regression tests**

Run:
- `pytest tests/historical_repro/test_historical_protocol_v19.py -q`
- `python -m py_compile tools/eval/eval_system_pipeline_v19.py tools/analysis/historical_best_pipeline_v19.py tools/analysis/run_multi_dataset_benchmark_v19.py`

- [ ] **Step 2: Run KIR50 tuned evaluations**

Run exact-history component-path evaluations for:
- `prototype` fixed
- `llm_verifier` with `--semantic_tuning_mode val_macro_f1`
- `fusion` with `--semantic_tuning_mode val_macro_f1`

- [ ] **Step 3: Compare whether tuned verifier/fusion improve end-to-end metrics**

Compare:
- `macro_f1`
- `known_intent_accuracy`
- `gate_oos_rejection`
- `gate_id_recall`
