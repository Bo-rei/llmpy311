# CLMSG pipeline audit

Audit base: `294d6f2` (2026-07-31). The working tree already contains the
uncommitted MOGB baseline integration. CLMSG uses a separate run root and does
not modify E0--E3, R1, M1, or MOGB artifacts.

## Reusable protocol components

| Concern | Authoritative implementation | CLMSG decision |
| --- | --- | --- |
| Dataset contract | `protocol_v2_textoir_v1` canonical data and registries | Reuse unchanged |
| Fixed splits | `gate/train.json`, `gate/val.json`, `gate/test.json` | Reuse unchanged |
| Runtime loading | `protocol_v2.gate.view_loader.load_gate_views` | Reuse |
| Frozen encoder | local `all-MiniLM-L6-v2` canonical cache | Cache-only; no implicit encoding |
| Cache validation | `experiments.runner._canonical_embedding_cache` and `_embedding_cache` | Reuse sample-ID and byte checks |
| Normalization | row-wise L2 normalization | Apply once inside CLMSG |
| OOS metrics | `evaluation.metrics.compute_binary_oos_metrics` | Reuse; add open-intent metrics |
| Provenance | registry/canonical/view/export hashes and atomic run directories | Reuse |
| Comparators | MOGB `all_runs.csv` for single, fixed-K2, and MOGB modes | Read only; never rerun |

The active source location is `src/protocol_v2/gate/clmsg.py`; orchestration is
kept in `scripts/experiments/run_clmsg.py`. No `src/s2c/` or parallel package
tree is introduced.

## Data flow and leakage boundary

```text
fixed registry + materialized views
    -> existing train_known as proper_train
    -> existing calibration_known as split-conformal calibration
    -> existing test_combined for one final evaluation
    -> validated frozen MiniLM cache
    -> L2-normalized local support model
```

The protocol already provides a disjoint Known-only validation split. It is
therefore used directly as calibration instead of re-splitting `train_known`
80/20. This preserves the TEXTOIR split, avoids creating another manifest, and
keeps every method on the same sample list. Both `train_known` and
`calibration_known` are checked to contain no OOS rows, and their sample-ID
sets must be disjoint. Held-out intents, native OOS, and all test rows are
excluded from fitting, local-scale estimation, and calibration.

## Milestone 1--3 interface

The first implementation deliberately contains only:

1. `knn_only`: kth-neighbour distance with a Known-calibration quantile;
2. `local_scale_knn`: minimum distance divided by the support point's
   same-intent local scale, with the natural support threshold 1;
3. `local_scale_conformal`: the same local-scale score converted to a global
   split-conformal p-value.

The manifold, label-entropy, cross-conformal, and hierarchical-calibration
modules are not implemented before Version C passes the registered gate.

## First experiment and expected cost

The first cell is StackOverflow, KIR 0.50, seed 13, frozen MiniLM, cosine
distance, `k=10`, and alphas 0.01/0.025/0.05/0.10. It reuses 6,000 proper-train,
1,000 calibration, and 6,000 test embeddings. Exact chunked matrix products
require roughly 42 million query--support distances plus per-intent local-scale
estimation; no encoder or GPU training is required. Seeds 42 and 87 run only if
seed 13 is numerically valid and Version C is not clearly dominated.

## Risks and controls

- A support sample may not be its own scale neighbour: same-intent scale
  estimation explicitly masks the diagonal.
- Calibration samples never enter the support model.
- Cosine and normalized-Euclidean distance are explicit configuration values.
- Small intents clip `k` to `n_intent - 1`; a singleton receives only `eps`.
- Exact normalized-distance search is chunked to bound memory.
- Alpha is a target Known false-rejection level, not a test-selected parameter.
- StackOverflow text is never written to predictions or Git-tracked results.
- All CLMSG files live below
  `artifacts/s2c/runs/protocol_v2_textoir_v1/clmsg_v1/`.

## New files for Milestone 1--3

- `src/protocol_v2/gate/clmsg.py`
- `scripts/experiments/run_clmsg.py`
- `configs/gates/clmsg.yaml`
- `tests/unit/test_clmsg.py`
- `docs/clmsg/CLMSG_PIPELINE_AUDIT.md`
- `docs/clmsg/CLMSG_RESEARCH_REPORT.md` (filled after the smoke decision)

