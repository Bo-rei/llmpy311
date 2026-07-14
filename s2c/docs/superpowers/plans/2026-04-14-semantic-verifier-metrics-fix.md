# Semantic Verifier Metrics Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the decoder-only semantic verifier evaluation so its validation metrics reflect yes/no token decisions correctly.

**Architecture:** Keep the change tightly scoped to the verifier training script and its tests. First add a failing test that reproduces the metrics bug, then patch the evaluation path so it maps logits to yes/no predictions using the configured token ids instead of treating raw vocabulary ids as class labels.

**Tech Stack:** Python 3.11, PyTorch, pytest

---

### Task 1: Add a failing verifier-metrics regression test

**Files:**
- Create: `tests/unit/test_train_semantic_verifier_v19.py`
- Modify: `tools/train/train_semantic_verifier_v19.py`
- Test: `tests/unit/test_train_semantic_verifier_v19.py`

- [ ] **Step 1: Write the failing test**

```python
def test_evaluate_prompt_model_counts_yes_no_via_token_ids():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_train_semantic_verifier_v19.py -q`
Expected: FAIL because the current implementation compares predictions against `1/0` instead of `yes_token_id/no_token_id`.

- [ ] **Step 3: Write minimal implementation**

```python
pred_is_yes = preds == yes_token_id
label_is_yes = labels == 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_train_semantic_verifier_v19.py -q`
Expected: PASS

### Task 2: Verify no regression in historical protocol tests

**Files:**
- Modify: `tools/train/train_semantic_verifier_v19.py`
- Test: `tests/historical_repro/test_historical_protocol_v19.py`

- [ ] **Step 1: Run targeted historical repro tests**

Run: `pytest tests/historical_repro/test_historical_protocol_v19.py -q`
Expected: PASS

- [ ] **Step 2: Run syntax check on modified training script**

Run: `python -m py_compile tools/train/train_semantic_verifier_v19.py`
Expected: PASS with no output
