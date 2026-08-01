# BRAK pilot report

## Question and contract

`Boundary-Risk-Aware Adaptive K Selection (BRAK)` selects a per-intent K from
`{1,2,3,4,5}` using only proper-train and Known calibration data.  Its score
combines self rejection, cross-intent calibration leakage, a bounded union
overlap/radius-expansion proxy, bootstrap assignment instability, and a
complexity penalty.  A candidate must retain calibration recall within 0.02 of
K=1 and improve the preregistered objective by at least 0.01; otherwise the
smallest safe K is retained.

No test OOS, native OOS, held-out intent, pseudo-OOS, or test-best selection is
used to choose K.  The pilot is independent of E0--E3, R1, MiniLM training,
and all MOGB artifacts.

## Pilot

| Field | Value |
| --- | --- |
| Protocol | `protocol_v2_textoir_v1` |
| Dataset | StackOverflow |
| KIR | 0.50 |
| Seeds | 42, 87, 100 |
| Candidate K | 1--5 per Known intent |
| Distance / radius | diagonal Mahalanobis / mean + std, lambda=1.0 |
| Selection source | proper train + Known calibration only |
| Runs | 3 selection runs, 15 fixed-K cells, 3 BRAK cells |
| Failures | 0 |

## Results

| Method | OOS F1 | F1-All | F1-K | Accuracy | Known Recall | False accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed K=1 | 0.8134 | 0.8010 | 0.7997 | 0.7890 | 0.8333 | 0.1992 |
| Fixed K=2 | 0.7062 | 0.7413 | 0.7448 | 0.7037 | 0.8756 | 0.3856 |
| Fixed K=3 | 0.6542 | 0.7121 | 0.7179 | 0.6697 | 0.8917 | 0.4592 |
| Fixed K=4 | 0.6481 | 0.7046 | 0.7103 | 0.6645 | 0.8893 | 0.4657 |
| Fixed K=5 | 0.6596 | 0.6952 | 0.6987 | 0.6662 | 0.8864 | 0.4506 |
| BRAK | 0.8134 | 0.8010 | 0.7997 | 0.7890 | 0.8333 | 0.1992 |

All 30 evaluated StackOverflow Known intents selected `K=1` (10 intents per
seed).  Average known-only candidate diagnostics increased in risk as K grew:

| Candidate K | Objective J | Calibration recall | Cross leakage | Union-risk proxy | Instability |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1879 | 0.8327 | 0.0206 | 0.0000 | 0.0000 |
| 2 | 0.5262 | 0.8497 | 0.0400 | 0.2613 | 0.2581 |
| 3 | 0.5774 | 0.8540 | 0.0570 | 0.2770 | 0.3099 |
| 4 | 0.6228 | 0.8530 | 0.0551 | 0.2831 | 0.4303 |
| 5 | 0.6518 | 0.8530 | 0.0571 | 0.2828 | 0.4995 |

## Decision

BRAK passes the safety requirement by refusing extra centers, but it does not
demonstrate a new adaptive-K gain on StackOverflow.  Fixed K>1 increases Known
recall while sharply increasing OOS false acceptance; the known-only risk
objective identifies that trade-off before test evaluation.  The predeclared
expansion gate is therefore **not met**: do not expand BRAK to all datasets/KIR
or claim it as a new main method.  The result is useful as a negative control
showing that a calibration-only risk selector can conservatively recover the
single-centroid operating point.

Evidence:

* `../artifacts/s2c/runs/protocol_v2_textoir_v1/brak_v1/BRAK_PROVENANCE.json`;
* `.../brak_v1/summaries/BRAK_PILOT_SUMMARY.tsv`;
* `.../brak_v1/summaries/BRAK_CANDIDATE_DIAGNOSTICS.tsv`;
* `.../brak_v1/summaries/BRAK_K_DISTRIBUTION.tsv`.
