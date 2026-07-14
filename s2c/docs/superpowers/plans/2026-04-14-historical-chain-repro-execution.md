# Historical Chain Reproduction Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover the historical-best CLINC150@KIR50 chain into a replay-validated bundle that can reproduce the archived result or isolate the first precise source of drift.

**Architecture:** Keep the top-level system fixed as Gate -> Router -> Expert, with `prototype` and `semantic verifier` documented and validated inside the Gate family only. Add a strict historical protocol layer, replay runner, and bundle generator around the existing code so the historical chain can be rerun under `data/v19` and classified by evidential status.

**Tech Stack:** Python 3.11, pytest, existing `tools/analysis/*` scripts, existing `tools/train/*` and `tools/eval/*` entrypoints, JSON/Markdown reports.

---

## File Structure

- Create: `tests/historical_repro/test_historical_protocol_v19.py`
  Purpose: lock the strict historical protocol in tests before changing production code.
- Modify: `tools/analysis/historical_best_pipeline_v19.py`
  Purpose: expose strict historical replay constants instead of only generalized multi-dataset defaults.
- Create: `tools/analysis/replay_historical_chain_v19.py`
  Purpose: run strict historical replay phases into a fresh output directory and capture stage-by-stage outcomes.
- Create: `tools/analysis/build_historical_repro_bundle_v19.py`
  Purpose: generate the bundle index, file status table, and replay summary from truth anchors plus replay outputs.
- Modify: `docs/prototype_gate_historical_best_code_map.md`
  Purpose: align the hand-written map with the replay-validated classification.
- Create: `docs/historical_repro_bundle/INDEX.md`
  Purpose: user-facing bundle entrypoint generated or refreshed from replay output.

### Task 1: Lock Strict Historical Protocol With Failing Tests

**Files:**
- Create: `tests/historical_repro/test_historical_protocol_v19.py`
- Test: `tests/historical_repro/test_historical_protocol_v19.py`

- [ ] **Step 1: Write the failing test**

```python
from tools.analysis.historical_best_pipeline_v19 import HISTORICAL_BEST_PIPELINE


def test_profile_exposes_strict_historical_data_root():
    strict = HISTORICAL_BEST_PIPELINE.strict_replay_defaults()
    assert strict["data_root"] == "data/v19"
    assert strict["known_intents_path"] == "data/v19/KNOWN_INTENTS.json"


def test_profile_exposes_historical_gate_encoder():
    strict = HISTORICAL_BEST_PIPELINE.strict_replay_defaults()
    assert strict["gate_encoder_path"] == "all-MiniLM-L6-v2"


def test_profile_exposes_historical_target_metrics():
    strict = HISTORICAL_BEST_PIPELINE.strict_replay_defaults()
    assert strict["target_metrics"]["macro_f1"] == 0.8121241036585573
    assert strict["target_metrics"]["overall_accuracy"] == 0.8677941443898891
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/historical_repro/test_historical_protocol_v19.py -q`
Expected: FAIL because `strict_replay_defaults()` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def strict_replay_defaults(self) -> Dict[str, object]:
    return {
        "data_root": "data/v19",
        "known_intents_path": "data/v19/KNOWN_INTENTS.json",
        "gate_encoder_path": "all-MiniLM-L6-v2",
        "target_metrics": {
            "macro_f1": 0.8121241036585573,
            "overall_accuracy": 0.8677941443898891,
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/historical_repro/test_historical_protocol_v19.py -q`
Expected: PASS

- [ ] **Step 5: Extend the same test file with archived data invariants**

```python
import json
from pathlib import Path


def test_historical_gate_test_split_counts_are_stable():
    data = json.loads(Path("data/v19/gate/test.json").read_text())
    id_count = sum(1 for row in data if row.get("intent") != "oos" and row.get("label") != 1)
    oos_count = len(data) - id_count
    assert len(data) == 5499
    assert id_count == 2249
    assert oos_count == 3250
```

- [ ] **Step 6: Run targeted test suite**

Run: `pytest tests/historical_repro/test_historical_protocol_v19.py -q`
Expected: PASS with all protocol anchor tests green.

- [ ] **Step 7: No commit step**

Current workspace is not a git repository; skip commit and record this in the final report instead of guessing.

### Task 2: Implement Strict Historical Replay Profile

**Files:**
- Modify: `tools/analysis/historical_best_pipeline_v19.py`
- Test: `tests/historical_repro/test_historical_protocol_v19.py`

- [ ] **Step 1: Write the failing test for artifact paths**

```python
def test_profile_exposes_strict_historical_artifact_paths():
    strict = HISTORICAL_BEST_PIPELINE.strict_replay_defaults()
    assert strict["reference_eval_results"].endswith(
        "pipeline_v19_phase3_proto_d_narrow_t085_eval/eval_results.json"
    )
    assert strict["frozen_detector_path"].endswith(
        "gate_l2_mix2_true_lambda_1p6/detector.json"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/historical_repro/test_historical_protocol_v19.py -q`
Expected: FAIL because strict artifact keys are not exposed yet.

- [ ] **Step 3: Add strict historical profile fields**

```python
def strict_replay_defaults(self) -> Dict[str, object]:
    return {
        "data_root": "data/v19",
        "known_intents_path": "data/v19/KNOWN_INTENTS.json",
        "gate_encoder_path": "all-MiniLM-L6-v2",
        "reference_eval_results": (
            "outputs/experiments/archive/sweeps/2026-03-23/"
            "pipeline_phase3_proto_eval/pipeline_v19_phase3_proto_d_narrow_t085_eval/"
            "eval_results.json"
        ),
        "frozen_detector_path": (
            "outputs/experiments/archive/sweeps/2026-03-23/"
            "gate_l2_mix2_train/gate_l2_mix2_true_lambda_1p6/detector.json"
        ),
        "target_metrics": {
            "macro_f1": 0.8121241036585573,
            "overall_accuracy": 0.8677941443898891,
            "known_intent_accuracy": 0.799466429524233,
            "oos_f1": 0.9096192078299434,
            "gate_oos_rejection": 0.9150769230769231,
            "gate_id_recall": 0.8599377501111605,
        },
    }
```

- [ ] **Step 4: Run tests to verify the profile is stable**

Run: `pytest tests/historical_repro/test_historical_protocol_v19.py -q`
Expected: PASS

- [ ] **Step 5: No commit step**

Current workspace is not a git repository; skip commit and record this in the final report instead of guessing.

### Task 3: Add Strict Historical Replay Runner

**Files:**
- Create: `tools/analysis/replay_historical_chain_v19.py`
- Test: `tests/historical_repro/test_historical_protocol_v19.py`

- [ ] **Step 1: Write the failing test for replay command construction**

```python
from tools.analysis.replay_historical_chain_v19 import build_strict_eval_command


def test_replay_eval_command_uses_historical_protocol():
    cmd = build_strict_eval_command(output_dir="outputs/tmp/historical_replay_eval")
    joined = " ".join(cmd)
    assert "--data_root data/v19" in joined
    assert "--gate_encoder_path all-MiniLM-L6-v2" in joined
    assert "pipeline_v19_phase3_proto_d_narrow_t085_eval" not in joined
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/historical_repro/test_historical_protocol_v19.py -q`
Expected: FAIL with import error because the replay runner does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from tools.analysis.historical_best_pipeline_v19 import HISTORICAL_BEST_PIPELINE


def build_strict_eval_command(output_dir: str) -> list[str]:
    strict = HISTORICAL_BEST_PIPELINE.strict_replay_defaults()
    return [
        "python",
        "tools/eval/eval_system_pipeline_v19.py",
        "--data_root",
        strict["data_root"],
        "--gate_encoder_path",
        strict["gate_encoder_path"],
        "--output_dir",
        output_dir,
    ]
```

- [ ] **Step 4: Expand the runner to emit replay manifests**

```python
manifest = {
    "mode": "strict_historical_replay",
    "data_root": strict["data_root"],
    "gate_encoder_path": strict["gate_encoder_path"],
    "target_metrics": strict["target_metrics"],
    "stages": [],
}
```

- [ ] **Step 5: Run tests to verify the replay runner contract**

Run: `pytest tests/historical_repro/test_historical_protocol_v19.py -q`
Expected: PASS

- [ ] **Step 6: Execute replay in report-only mode first**

Run: `python tools/analysis/replay_historical_chain_v19.py --mode report_only --output_root outputs/reports/historical_replay_20260414`
Expected: writes a strict replay manifest without modifying legacy artifacts.

- [ ] **Step 7: No commit step**

Current workspace is not a git repository; skip commit and record this in the final report instead of guessing.

### Task 4: Build Replay-Validated Historical Bundle

**Files:**
- Create: `tools/analysis/build_historical_repro_bundle_v19.py`
- Create: `docs/historical_repro_bundle/INDEX.md`
- Modify: `docs/prototype_gate_historical_best_code_map.md`
- Test: `tests/historical_repro/test_historical_protocol_v19.py`

- [ ] **Step 1: Write the failing test for bundle metadata**

```python
from tools.analysis.build_historical_repro_bundle_v19 import classify_file_status


def test_bundle_classifies_frozen_baseline_wrapper_as_mainchain():
    assert classify_file_status("tools/analysis/run_prototype_gate_frozen_baseline_v19.py") in {
        "VERIFIED_MAINCHAIN",
        "FROZEN_DEPENDENCY",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/historical_repro/test_historical_protocol_v19.py -q`
Expected: FAIL with import error because the bundle builder does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def classify_file_status(relpath: str) -> str:
    if relpath in {
        "tools/analysis/run_prototype_gate_frozen_baseline_v19.py",
        "tools/eval/eval_system_pipeline_v19.py",
    }:
        return "VERIFIED_MAINCHAIN"
    return "REFERENCE_ONLY"
```

- [ ] **Step 4: Add bundle index generation**

```python
index = """# Historical Reproduction Bundle

- truth anchors
- strict replay status
- file classification table
"""
```

- [ ] **Step 5: Run tests to verify bundle metadata**

Run: `pytest tests/historical_repro/test_historical_protocol_v19.py -q`
Expected: PASS

- [ ] **Step 6: Generate the bundle**

Run: `python tools/analysis/build_historical_repro_bundle_v19.py --output_dir docs/historical_repro_bundle`
Expected: creates or refreshes `docs/historical_repro_bundle/INDEX.md` and companion JSON/Markdown summaries under `outputs/reports`.

- [ ] **Step 7: No commit step**

Current workspace is not a git repository; skip commit and record this in the final report instead of guessing.

### Task 5: Execute Strict Historical Replay And Classify Drift

**Files:**
- Modify: `tools/analysis/replay_historical_chain_v19.py`
- Modify: `tools/analysis/build_historical_repro_bundle_v19.py`
- Modify: `docs/prototype_gate_historical_best_code_map.md`
- Test: `tests/historical_repro/test_historical_protocol_v19.py`

- [ ] **Step 1: Extend replay from report-only to stage execution**

```python
stages = [
    "truth_freeze",
    "data_validation",
    "gate_family_replay",
    "router_replay",
    "expert_replay",
    "pipeline_eval",
]
```

- [ ] **Step 2: Run the replay**

Run: `python tools/analysis/replay_historical_chain_v19.py --mode full --output_root outputs/reports/historical_replay_20260414`
Expected: creates a fresh replay directory, executes strict historical stages, and records stage-local status without overwriting old artifacts.

- [ ] **Step 3: Compare replay metrics to truth anchors**

Run: `python tools/analysis/build_historical_repro_bundle_v19.py --output_dir docs/historical_repro_bundle --replay_root outputs/reports/historical_replay_20260414`
Expected: bundle index contains the archived metrics, replay metrics, metric deltas, and the first diverging stage if exact reproduction fails.

- [ ] **Step 4: Re-run tests**

Run: `pytest tests/historical_repro/test_historical_protocol_v19.py -q`
Expected: PASS

- [ ] **Step 5: Record final evidence sweep**

Run: `python tools/analysis/validate_historical_best_chain_v19.py`
Expected: refreshed mother-chain report still points to valid truth anchors and complements the new replay bundle.

- [ ] **Step 6: No commit step**

Current workspace is not a git repository; skip commit and record this in the final report instead of guessing.

## Self-Review

Spec coverage:
- Recover-and-replay objective is covered by Tasks 2, 3, and 5.
- File classification and research bundle output are covered by Task 4.
- Gate-family framing is preserved by making replay and bundle logic treat `prototype` and `semantic verifier` as Gate internals only.

Placeholder scan:
- No `TODO`, `TBD`, or deferred "write tests later" placeholders remain.
- All tasks include exact file paths, commands, and expected outcomes.

Type consistency:
- `strict_replay_defaults()` is used consistently as the profile contract.
- `build_strict_eval_command()` and `classify_file_status()` keep the same names everywhere they appear.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-14-historical-chain-repro-execution.md`. Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

The user already requested execution in this session, so continue with Inline Execution unless blocked.
