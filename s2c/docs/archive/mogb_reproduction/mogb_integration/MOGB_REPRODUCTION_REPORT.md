# MOGB integration, strict reproduction, and fair component closeout

## Scope

This report combines three explicitly separated evidence families. First, it
records the strict StackOverflow single-cell execution of the pinned MOGB
training/evaluation path. Second, it closes the protocol-aligned frozen-MiniLM
component matrices. Third, it records BRAK applied to the initial and trained
MOGB BERT representations. None of these changes the s2c Gate, E2/E3/R1
artifacts, or the Gate--Router--Expert pipeline.

The strict run is a reproduction attempt, not an automatic SOTA claim. The
official MOGB paper reports `Acc=88.67`, `F1-All=87.49`, `F1-U=89.71`, and
`F1-K=87.27` for its StackOverflow KIR=50% setting. The local exact-contract
run uses the pinned source, local BERT, the audited 20,000-row snapshot, and
the registered seed diagnostics; it is reported separately from the fair
MiniLM matrix because the metric and data contracts differ.

## Strict single-cell MOGB reproduction

### Contract

* Run id: `mogb_exact_reproduction_v1`.
* Dataset: StackOverflow; 20 labels x 1,000 rows; source train/dev/test
  counts 12,000/2,000/6,000.
* Known split: KIR=0.50, seed=0; ten Known and ten held-out intents are
  recorded in `audit/known_intents.json`.
* Backbone: local `bert-base-uncased`; max sequence length 45; batch sizes
  128/64; learning rates `2e-5` and `1e-4`; max epochs 100; patience 10.
* Granular-ball parameters: train purity .90, get-ball purity 1.00,
  select-ball purity .90, minimum sizes 10/5/10, step 5.
* Both seed contracts were run: `official_fixed` preserves the upstream
  pre-Data seed 100 before Data resets seed 0; `unified_zero` sets every
  recorded source to zero. They produced the same checkpoint and metrics.

The source snapshot hash is
`b8410dbc8677d4c57a578d68343b9b8ae9dda54fee9ee8eba3f0a142acd9b397` and the
selected checkpoint hash is
`e43553d9d52a75277726a5d8256b9cca93df43dc1048cd694b9e8e26679137f0`.
The complete provenance is
`../artifacts/s2c/external/mogb_exact_reproduction_v1/audit/MOGB_EXACT_PROVENANCE.json`.

### Result and paper gap

| Seed contract | Acc | F1-All | F1-U | F1-K | Best dev Acc | Epochs | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| official_fixed | 75.1667 | 68.3502 | 79.9676 | 67.1884 | 91.6 | 38 | not_reproduced |
| unified_zero | 75.1667 | 68.3502 | 79.9676 | 67.1884 | 91.6 | 38 | not_reproduced |
| paper reference | 88.6700 | 87.4900 | 89.7100 | 87.2700 | -- | -- | published reference |

The strict gaps are respectively -13.5033, -19.1398, -9.7424 and -20.0816
percentage points for Acc, F1-All, F1-U and F1-K. The result is therefore
`not_reproduced`, not `approximate` or `SOTA`.

The two seed contracts being byte-identical rules out the simple seed-order
explanation. The run still has a large Known-recall loss (51.5333%) and its
selected final balls include many tiny 10--32 sample balls, so the report
keeps model/data-contract, ball-selection and evaluation-contract differences
as diagnostic hypotheses rather than silently tuning them after seeing test
metrics. No additional official-BERT sweep is authorized in this stage.

### Compatibility boundary

The pinned checkout is preserved under `third_party/mogb_official` and was not
edited. The isolated runner repairs modern PyTorch graph lifetime and device
handling while preserving the official CE, epoch-end feature extraction,
GBNR split, nearest sub-centroid loss, mean-radius balls and nearest-ball
evaluation sequence. Its numeric compatibility test is a fixed-centroid
formula check, not a proof of byte-for-byte equivalence of every autograd
operation. This limitation is recorded explicitly in the mode manifests.

## BRAK on MOGB representations

The separate `brak_mogb_representation_v1` run evaluates fixed K=1..5 and the
Known-only BRAK selector on the same StackOverflow KIR=.50/seed=0 split. It
does not use test OOS for selection and does not change the MOGB checkpoint.

| Representation | BRAK OOS F1 | BRAK F1-All | BRAK F1-K | Accuracy | Selected K |
| --- | ---: | ---: | ---: | ---: | --- |
| Frozen MiniLM | 0.8450 | 0.8213 | 0.8189 | 0.8208 | all 10 intents K=1 |
| MOGB initial BERT | 0.2460 | 0.0663 | 0.0483 | 0.1170 | all 10 intents K=1 |
| MOGB trained BERT | 0.0000 | 0.0228 | 0.0251 | 0.0282 | 8 K=1, 2 K=2 |

On the trained MOGB representation, fixed K=2--5 improve the raw OOS F1
slightly from a collapsed K=1 baseline, but all absolute scores remain poor
and BRAK's Known-only selector stays conservative. This is a negative control
and representation-transfer diagnostic, not evidence for a new adaptive-K
method. The full 18-row summary and provenance are under
`results/mogb_exact_reproduction/brak_mogb_representation/` and
`../artifacts/s2c/external/mogb_exact_reproduction_v1/brak_mogb_representation/`.

## Strict Banking77 follow-up

To distinguish a StackOverflow-specific contract issue from a general failure
of the modernized official path, one additional single cell was run with the
official example setting `banking`, KIR=.75, seed=0. It uses the local
13,083-row Banking77 snapshot and the same pinned BERT/compatibility runtime;
it is not merged into the protocol_v2 MiniLM tables.

| Dataset | KIR | Acc | F1-All | F1-U | F1-K | Known Recall | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| StackOverflow | .50 | 75.1667 | 68.3502 | 79.9676 | 67.1884 | 51.5333 | not reproduced |
| Banking77 | .75 | 57.0779 | 59.1627 | 53.1049 | 59.2671 | 43.7069 | not reproduced |

The Banking cell completed 54 epochs and selected its checkpoint by dev
accuracy. Its descriptive gaps to the paper's Banking reference are
`-23.5021/-22.3573/-27.9351/-22.2629` pp for Acc/F1-All/F1-U/F1-K, but the
paper reference is not an exact same-KIR comparison. The two cells therefore
support the conservative statement
`official_code_not_reproduced_under_available_contract`; they do not justify
a larger official-BERT sweep.

## Modes

* `mogb_official_reproduction`: the strict StackOverflow single-cell attempt
  completed under an isolated compatibility layer but is classified
  `not_reproduced` against the published reference table.
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

## Historical non-strict compatibility aggregate

Before the strict single-cell run, an isolated modernized compatibility
aggregate completed five seeds for StackOverflow and Banking77. Its official
format F1-All means were 40.7243 and 19.2843, respectively. Those values are
retained under `../artifacts/s2c/external/mogb_official_converged_v1/` as
non-strict compatibility evidence. They use a different local data/split
contract and are not combined with the strict single-cell result or the
frozen-MiniLM table.

The earlier one-epoch engineering smoke remains archived under
`../artifacts/s2c/external/mogb_official_modernized_smoke_v1/`; its output was
never treated as a converged metric.

## Interpretation rule

Published MOGB numbers are a descriptive reference until the original sample
draw, Known intent list, split and metric implementation are independently
aligned.  A MiniLM-fair result can establish a component comparison; it cannot
be labeled an official MOGB reproduction or an s2c SOTA claim.

## Stop/continue decision

The strict single-cell MOGB attempt, the 540-cell frozen-MiniLM OFAT matrix,
the 180-cell fixed-K mean-radius comparison and the BRAK representation
diagnostic are complete. They are sufficient to separate three claims:

1. fixed/adaptive partition behavior under Frozen MiniLM;
2. the result of the pinned official training path under a modern compatibility
   layer; and
3. whether Known-only BRAK can safely select extra centers on the resulting
   representations.

They do not authorize a new adaptive-K method, a full Pipeline claim, or a
claim that the local strict run reproduces the published MOGB numbers. ADB and
DA-ADB now each have one completed modernized compatibility cell on
StackOverflow/KIR=.50/seed=0 (`F1-open=89.4712` and `90.8978`); these are
independent boundary references, not strict protocol_v2 or multi-seed SOTA
results. DCLOOS's official README identifies an external SQuAD-derived
`squad.tsv`; the isolated runner records the byte-identical
`squad.tsv` -> `squad_placeh.tsv` loader-name assumption. The default-budget
cell exceeded the three-hour ceiling; a separately registered reduced-budget
cell reached upstream test evaluation and recovered metrics from 5,700
prediction rows after a JSON-serialization defect. That recovery is
compatibility evidence only, not a strict default or paper reproduction. It
must not be replaced by ADB/DA-ADB or by protocol test OOS.
