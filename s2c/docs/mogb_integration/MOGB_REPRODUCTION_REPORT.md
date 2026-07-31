# MOGB integration and frozen-MiniLM component closeout

## Scope

This stage closes the protocol-aligned frozen-MiniLM MOGB OFAT sweep. It does
not alter the s2c Gate, MiniLM training objectives, E2/E3 artifacts, or the
Gate--Router--Expert pipeline. The registered sweep completed 540/540 cells
with zero failures across three datasets, three KIR values, five seeds and
twelve one-factor-at-a-time partition/boundary variants.

## Modes

* `mogb_official_reproduction`: currently blocked at the legacy dependency and
  data-contract preflight; no fabricated number is emitted.
* `mogb_minilm`: default MOGB partitioning with Euclidean distance and mean
  radius; this is the frozen-MiniLM reference, not the official BERT model.
* `get_090`: best partition-only setting; purity threshold `0.90` on the
  frozen protocol_v2 MiniLM cache.
* `default_mean_std`: the default partition with Euclidean distance and the
  s2c `mean + std` radius. The ablation also scans diagonal Mahalanobis and
  mean-radius boundary combinations.

The protocol-aligned fair matrix also records single-centroid,
random-balanced, and fixed-K=2 controls. All methods share one registry, one
view, one embedding cache and one evaluator.

## Current status

The official source and current pipeline audit are complete. The frozen-MiniLM
OFAT sweep completed 540/540 cells with zero failures. The default MOGB
reference is Euclidean + mean radius, the best partition-only setting is
`get_090`, and Euclidean + mean_std is the strongest frozen-representation
setting on OOS F1. That gain does not remove the Known Recall / F1-All trade-off
against the single-centroid control.

The reproducible entrypoints are `scripts/experiments/reproduce_mogb_original.py`
for the upstream preflight, `scripts/experiments/run_mogb_fair.py` for one
protocol-aligned cell, `scripts/experiments/run_mogb_sweep.py` for a deterministic
dry-run/resume sweep, and `scripts/experiments/aggregate_mogb_results.py` for
lightweight aggregation.  The active protocol materializes seeds
`13, 42, 87, 100, 123`; seed `0` is not silently regenerated.

### Aggregate closeout summary

| Configuration | Mean OOS F1 | Delta vs default MOGB | Delta vs single centroid | Mean F1-All | Known recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default MOGB: Euclidean + mean radius | 0.7339 | 0.0000 | -0.0525 | 0.4626 | 0.3121 |
| Best partition-only setting: `get_090` | 0.7474 | +0.0135 | -0.0391 | 0.5263 | 0.3777 |
| Euclidean + mean_std radius | 0.7865 | +0.0526 | +0.0001 | 0.6782 | 0.5756 |
| Diagonal Mahalanobis + mean radius | 0.7283 | -0.0056 | -0.0582 | 0.4282 | 0.2820 |
| Diagonal Mahalanobis + mean_std radius | 0.7774 | +0.0435 | -0.0090 | 0.6392 | 0.5221 |
| Single centroid control | 0.7864 | +0.0525 | 0.0000 | 0.7830 | 0.8350 |

The largest frozen-representation improvement comes from the radius rule, not
from the recursive partition thresholds. `mean_std` raises OOS F1 by 5.26
points over the default MOGB reference and matches the single-centroid OOS F1
to within rounding, but F1-All remains 10.48 points lower and Known recall
remains 25.93 points lower. The best partition-only setting, `get_090`,
improves over default MOGB but does not recover the open-intent trade-off.

### Fixed K versus adaptive partition under the same boundary

To remove the remaining partition/boundary confound, a second registered stage
holds Frozen MiniLM, L2 normalization, Euclidean distance, mean-distance radius
and nearest-ball inference constant. It evaluates fixed per-intent K=1/3/4 in
135 new cells and reuses 45 exactly equivalent fixed-K2 cells after per-cell
manifest and input-hash validation. All 180 fixed-K cells are paired with the
45 adaptive `mogb_minilm` references.

| Partition | Mean OOS F1 | Mean F1-All | Known recall | False accept | Delta vs adaptive OOS F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed K=1 | 0.7808 | 0.6616 | 0.5385 | 0.0521 | +0.0469 |
| Fixed K=2 | 0.7590 | 0.6230 | 0.4998 | 0.0735 | +0.0251 |
| Fixed K=3 | 0.7477 | 0.5922 | 0.4671 | 0.0770 | +0.0138 |
| Fixed K=4 | 0.7482 | 0.5812 | 0.4547 | 0.0694 | +0.0143 |
| Adaptive MOGB balls | 0.7339 | 0.4626 | 0.3121 | 0.0090 | 0.0000 |

Fixed K=1 beats the adaptive partition in all 45 paired protocol cells. The
adaptive partition also trails fixed K=2 in 39/45 cells. It rejects OOS very
aggressively, but its mean-radius balls reject far more Known samples. CLINC150
selects K=1 descriptively in all 15 fixed-K cells and StackOverflow does so in
14/15; Banking77 distributes its descriptive winners across K=1, K=2 and K=4.
These results rule out dynamic granularity alone as the explanation for the
published MOGB advantage. They do not evaluate MOGB's hierarchical BERT
representation learning, which remains the principal untested component.

Machine-readable evidence is under
`../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_fixed_k_mean_ablation_v1/summary/`.
The verifier confirms 180/180 fixed-K rows, 45/45 adaptive references, no input
hash mismatch and no test-based K selection.

### Full MiniLM-fair matrix (mean over five seeds and three KIR values)

| Dataset | Method | OOS F1 | F1-All | Known Recall | False accept |
| --- | --- | ---: | ---: | ---: | ---: |
| CLINC150 | Single centroid | 0.8830 | 0.8015 | 0.7859 | 0.0832 |
| CLINC150 | Fixed K=2 | 0.8839 | 0.7985 | 0.7510 | 0.0616 |
| CLINC150 | MOGB partition + s2c boundary | 0.8389 | 0.6479 | 0.5387 | 0.0221 |
| Banking77 | Single centroid | 0.7042 | 0.7400 | 0.8511 | 0.3489 |
| Banking77 | Fixed K=2 | 0.7352 | 0.7584 | 0.8253 | 0.2849 |
| Banking77 | MOGB partition + s2c boundary | 0.7486 | 0.6486 | 0.5324 | 0.0386 |
| StackOverflow | Single centroid | 0.7721 | 0.8074 | 0.8679 | 0.2583 |
| StackOverflow | Fixed K=2 | 0.6461 | 0.7467 | 0.8671 | 0.4277 |
| StackOverflow | MOGB partition + s2c boundary | 0.7447 | 0.6213 | 0.4952 | 0.0190 |

The full matrix does not support one universal MOGB-style winner. The adaptive
partition plus s2c boundary often raises binary OOS F1 by accepting far fewer
OOS samples, but it also rejects a large fraction of Known samples; its
F1-All and Known Recall are consistently below the single-centroid control.
These are component and operating-point findings, not an official MOGB paper
reproduction or an SOTA claim.

The complete machine-readable evidence is under
`../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_baseline_v1/summary/`:
`all_runs.csv`, `mean_std.csv`, `significance_tests.csv` and
`per_intent_analysis.csv`. The Git-tracked lightweight export is
`s2c/results/mogb/fair_matrix.csv`; its SHA256 is recorded in
`s2c/results/MANIFEST.csv`.

## Interpretation rule

Published MOGB numbers are a descriptive reference until the original sample
draw, Known intent list, split and metric implementation are independently
aligned.  A MiniLM-fair result can establish a component comparison; it cannot
be labeled an official MOGB reproduction or an s2c SOTA claim.

## Stop/continue decision

The registered 540-cell frozen-MiniLM OFAT matrix and 180-cell fixed-K
mean-radius comparison are complete. They are sufficient for the component
conclusions above, but they do not authorize a new adaptive-K
method, a full Pipeline claim, or a claim that the adapter is official MOGB.
The official BERT reproduction remains `audited_not_reproduced` because the
pinned source is not runnable under the current legacy contract. Any future
official reproduction must use an isolated modernized copy and a separately
audited original data/split contract.
