# StackOverflow Gate contract audit

## Scope

This was a read-only audit of one frozen `protocol_v2_textoir_v1` cell:
StackOverflow, KIR `0.50`, seed `42`, diagonal Mahalanobis, `mean_std`,
`lambda=1.0`, and `K={1,2}`. It did not retrain MiniLM, rebuild canonical
data, or write to E2/E3/R1 artifact roots.

The local TEXTOIR-compatible snapshot was also evaluated through the existing
TEXTOIR ADB compatibility wrapper. That run reported `F1-K=83.7033`,
`F1-U=85.6422`, `F1-All=83.8796`, and `Accuracy=84.43`; the wrapper left the
TEXTOIR worktree unchanged. The result is a data/adapter smoke check, not a
claim of a fair MOGB comparison.

## Defects separated

1. `MultiSphereOOSDetector` historically selected the raw-distance nearest
   sphere and divided by that sphere's radius. The new opt-in
   `acceptance_mode="normalized_union"` computes all distance/radius ratios,
   accepts when any sphere contains the point, and reports the minimum ratio.
   The default remains `nearest_sphere` so frozen E2/E3 numbers are unchanged.
2. Split embedding cache hits now validate the recorded embedding bytes and
   compare the cache values with the current ordered view sample IDs projected
   from the canonical cache. E3 additionally validates actual cache bytes and
   the complete train/test projection against canonical rows.
3. Active E2/E3 still use per-intent `class_centroid_mixture`; the legacy
   global `center_mode="kmeans"` path is not the source of the active E2 K2
   results and remains outside this contract change.

## Single-cell evidence

The diagnostic JSON is:

`../artifacts/s2c/runs/protocol_v2_textoir_v1/detector_contract_audit_v1/stackoverflow__kir_0.50__seed_42.json`

The historical detector exactly reproduced the stored E2 metrics:

| K | E2 OOS F1 | Recomputed OOS F1 | E2 ID Recall | Recomputed ID Recall |
|---:|---:|---:|---:|---:|
| 1 | 0.819705 | 0.819705 | 0.838000 | 0.838000 |
| 2 | 0.688332 | 0.688332 | 0.883333 | 0.883333 |

The opt-in normalized union diagnostic changed the operating geometry but did
not rescue K2: OOS F1 was `0.810849` for K1 and `0.693225` for K2. This is a
contract diagnostic, not a replacement result for E2.

## Decision

The StackOverflow snapshot and current E2 cache are not implicated as a simple
data corruption problem: the TEXTOIR ADB smoke is close to the repository
reference, and the active E2 cache is content- and sample-order aligned.
Before the MiniLM Representation Adaptation Study (`M1`), retain these fixes,
use the historical nearest-sphere mode only for legacy reproduction, and make
any new Gate score mode explicit in the new M1 run manifest. Do not rerun E2 or
E3, and do not infer a universal multi-center benefit from this audit.

## Verification

- targeted unit tests: 24 passed;
- full unit suite: 240 passed;
- integration tests: 8 passed;
- smoke tests: 3 passed;
- Ruff, `compileall`, and `git diff --check`: passed;
- E2 source manifests and metrics remained unchanged during the audit.
