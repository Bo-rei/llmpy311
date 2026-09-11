# Strict Known-only full-pipeline ablation

## Conclusion

The cascade ablation was rerun as a fresh current-H1 full-pipeline evaluation:
3 datasets × 3 KIR values × 5 variants × 3 seeds, 135/135 CUDA units. Every
variant uses only Known training rows. All auxiliary thresholds are calibrated
from Known validation rows at 90% Known coverage. No real OOS, pseudo-OOS or
test row is used for training or selection; test rows are loaded only after the
calibration phase. This is a current H1 result, not a strict H0 reconstruction
of the historical cascade.

The previous paper-shaped runner is not used for this table because its
auxiliary thresholds were selected with validation full-macro-F1. The runner
and complete machine-readable evidence are:

- [runner](../../tools/eval/run_known_only_full_pipeline_ablation.py)
- [manifest](../../results/analysis/trainable_full_pipeline_ablation_known_only/MANIFEST.json)
- [per-seed results](../../results/analysis/trainable_full_pipeline_ablation_known_only/per_seed.csv)
- [summary](../../results/analysis/trainable_full_pipeline_ablation_known_only/summary.csv)

## Table values

Each entry is `Acc / OOS F1` in percent, reported as mean±std over seeds 13,
42 and 87. The corrected Cascade-MiniLM uses an independent Frozen MiniLM
Gate and Known-only threshold, so its OOS decision is not copied from Ours.

| KIR | Dataset | Ours | Frozen MiniLM | Without Gate | Cascade-MiniLM | Cascade-SmolLM |
|---:|---|---:|---:|---:|---:|---:|
| .25 | CLINC150 | 91.18±0.65 / 95.18±0.48 | 84.39±1.69 / 90.52±0.97 | 54.15±5.00 / 61.96±5.78 | 85.58±1.40 / 90.52±0.97 | 68.34±2.80 / 77.25±2.36 |
| .50 | CLINC150 | 88.23±0.87 / 92.13±0.81 | 80.18±0.36 / 84.40±0.50 | 66.13±1.65 / 66.33±2.38 | 82.25±0.45 / 84.40±0.50 | 64.08±1.50 / 64.36±2.20 |
| .75 | CLINC150 | 85.58±1.07 / 86.80±1.55 | 76.08±0.49 / 72.72±0.80 | 71.52±1.64 / 60.93±3.50 | 78.53±0.19 / 72.72±0.80 | 70.42±1.52 / 60.85±3.40 |
| .25 | StackOverflow | 91.76±0.87 / 95.50±0.96 | 86.08±3.05 / 91.60±2.21 | 51.41±6.52 / 56.22±7.94 | 86.69±3.18 / 91.60±2.21 | 29.38±1.63 / 21.41±2.03 |
| .50 | StackOverflow | 86.33±1.59 / 89.23±2.04 | 75.51±3.52 / 76.83±5.51 | 64.50±0.26 / 58.10±0.58 | 76.78±3.52 / 76.83±5.51 | 45.44±1.52 / 18.14±2.32 |
| .75 | StackOverflow | 82.28±0.96 / 75.94±1.93 | 76.63±0.92 / 63.51±3.78 | 75.00±0.86 / 54.01±4.06 | 78.12±0.97 / 63.51±3.78 | 62.80±1.18 / 14.93±2.86 |
| .25 | BANKING77-OOS | 81.66±3.17 / 89.07±2.16 | 70.74±6.36 / 80.98±5.00 | 62.04±3.30 / 73.64±3.03 | 71.08±6.43 / 80.98±5.00 | 32.39±1.28 / 40.74±1.91 |
| .50 | BANKING77-OOS | 75.69±4.68 / 83.55±3.85 | 61.80±2.22 / 70.48±2.21 | 66.05±4.37 / 74.34±4.21 | 62.63±2.15 / 70.48±2.21 | 40.43±0.81 / 43.52±0.79 |
| .75 | BANKING77-OOS | 73.28±1.51 / 79.91±1.67 | 61.13±1.70 / 65.72±2.51 | 67.58±1.24 / 73.03±1.44 | 62.45±1.66 / 65.72±2.51 | 48.99±0.52 / 48.60±1.03 |

## Ours configuration lock

| Dataset | KIR | Trainable Gate configuration | Selection source |
|---|---:|---|---|
| CLINC150 | .25 | K=2, Euclidean, mean+0 std, nearest-sphere, ratio fusion (w=1), threshold 2.271481 | Known-only coverage-90 lock |
| CLINC150 | .50 | K=2, Euclidean, intent quantile .70, nearest-sphere, inverse-margin fusion (w=.5), threshold 2.403394 | Known-only coverage-90 lock |
| CLINC150 | .75 | K=1, Euclidean, mean+2 std, nearest-sphere, max-ratio fusion (w=2), threshold 1.373377 | Known-only coverage-90 lock |
| StackOverflow | .25/.50/.75 | K=1, Euclidean, mean+1.5 std, normalized-union, no fusion, threshold .90 | Fixed Known-only contract; checkpoint manifest records Known-only validation selection |
| BANKING77-OOS | .25 | K=4, Euclidean, mean+0 std, nearest-sphere, inverse-margin fusion (w=1), threshold 9.252483 | Known-only coverage-90 lock |
| BANKING77-OOS | .50 | K=1, Euclidean, mean+0 std, nearest-sphere, inverse-margin fusion (w=1), threshold 5.090498 | Known-only coverage-90 lock |
| BANKING77-OOS | .75 | K=1, Euclidean, mean+0 std, nearest-sphere, inverse-margin fusion (w=1), threshold 10.395770 | Known-only coverage-90 lock |

Frozen MiniLM uses the same per-cell boundary geometry with a frozen encoder
and a threshold recalibrated from Known validation only. Without Gate uses
downstream intent-confidence rejection with a Known-validation threshold.
Cascade-MiniLM uses the frozen MiniLM representation for both its Gate and its
logistic Router/Expert heads. Cascade-SmolLM uses the SmolLM prototype Gate
and the fixed SmolLM downstream.

## Ours diagnostic metrics

| Dataset | KIR | Known F1 | OOS F1 | Acc | Known Recall | False Acceptance |
|---|---:|---:|---:|---:|---:|---:|
| CLINC150 | .25 | 78.41±1.63 | 95.18±0.48 | 91.18±0.65 | 87.95±1.02 | 6.32±1.02 |
| CLINC150 | .50 | 82.98±0.79 | 92.13±0.81 | 88.23±0.87 | 89.27±1.08 | 8.25±1.16 |
| CLINC150 | .75 | 84.71±0.64 | 86.80±1.55 | 85.58±1.07 | 89.71±0.16 | 10.89±2.84 |
| StackOverflow | .25 | 80.27±0.37 | 95.50±0.96 | 91.76±0.87 | 83.23±0.38 | 3.49±1.96 |
| StackOverflow | .50 | 83.13±1.50 | 89.23±2.04 | 86.33±1.59 | 83.08±1.03 | 5.75±3.79 |
| StackOverflow | .75 | 85.04±1.06 | 75.94±1.93 | 82.28±0.96 | 82.67±0.76 | 6.94±3.56 |
| BANKING77-OOS | .25 | 53.64±3.79 | 89.07±2.16 | 81.66±3.17 | 88.19±1.47 | 18.37±3.67 |
| BANKING77-OOS | .50 | 61.86±4.26 | 83.55±3.85 | 75.69±4.68 | 90.40±0.29 | 25.82±5.97 |
| BANKING77-OOS | .75 | 66.34±0.80 | 79.91±1.67 | 73.28±1.51 | 90.29±0.08 | 29.58±2.48 |

## Interpretation boundary

### Final-decision audit

The evaluator includes Gate-accepted OOS utterances as OOS false negatives.
The current Router and Experts only output Known labels; changing their
assignment of an accepted OOS from one Known label to another does not change
its false-negative status. A downstream rejection would change OOS F1, but
the evaluated Ours and Cascade-MiniLM variants have no such rejection stage.
The manuscript's claim that downstream inputs are all genuinely Known was
incorrect and has been corrected to include misaccepted OOS inputs.

The corrected three-seed Ours OOS F1 agrees with the Known-only source lock
for all nine cells up to the recorded replay precision. It is now directly
comparable to the three-seed ablation rows under the same H1 data family.

The main table also retains older Known F1/Accuracy values alongside newer
CLINC150/StackOverflow OOS F1 values, and retains older Banking OOS values.
It therefore requires a consistent full-pipeline, three-seed result source
before the main and ablation tables can be treated as matched evidence.

Under the corrected contract, Ours is higher than Frozen MiniLM,
Cascade-MiniLM, Without Gate and Cascade-SmolLM in OOS F1 and Accuracy in
all nine dataset×KIR settings. The equality between Ours and
Cascade-MiniLM has disappeared because the latter now has its own frozen
MiniLM Gate. The remaining equality between Frozen MiniLM and Cascade-MiniLM
on OOS F1 is expected: they share the same frozen Gate, while their
downstream classifiers affect Known classification and Accuracy.
