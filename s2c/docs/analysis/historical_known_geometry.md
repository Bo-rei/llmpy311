# Strict Known-only geometry search

2026-09-11: the original checkpoint geometry search finished all 27 validation
units. Real CLINC/.75/seed13 replay on 2240 Known validation rows exactly
reproduced recorded utility 0.8973214285714286. Synthetic scoring replay and
training freeze-contract tests passed. The eleven-recipe training search is
running on CUDA; final test evaluation has not started.

Base commit: d0b10e267a3ca41c2130b804e26900c1e5c53077.
Runner: `scripts/experiments/search_historical_known_geometry.py`.
Protocol: historical_v19_paper_main, H1 Gate-only.

All 27 dataset/KIR/seed cells use existing Known-validation-selected checkpoints.
Search uses only Known train and validation rows, with no pseudo-OOS.
Utility is correct accepted fraction minus four times wrong accepted fraction;
rejecting a Known sample gives zero utility. This is a surrogate for selective
classification, not an estimate of OOS F1. Stable candidate order breaks ties.

The search covers K=1..5, Euclidean/diagonal Mahalanobis, mean/std lambda,
per-center and per-intent quantiles, nearest/union acceptance, threshold,
and fusion with nearest/second-nearest distinct-intent distance ratio.
Per-intent quantile uses the nearest-center distance of that intent's training
samples. This calibration stage does not read test files.

Outputs: `results/analysis/historical_known_geometry/` contains candidate metrics,
selection locks and a manifest. No representations or downstream components
are retrained in this stage. New representation search and final evaluation
remain outstanding. This is a new selection objective, not a rerun of a completed
OOS-validation search.

Training extension: `scripts/experiments/search_historical_known_training.py`
screens eleven fixed recipes on seed42 for all nine dataset/KIR settings,
then expands the two recipes with highest Known-validation macro F1 to seeds
13 and 87. Recipes vary last 1/2/4/6 layers, projection bypass/128/256,
temperature, intra/inter weights, margin, LR, and checkpoint epoch (1..9).
No test file is opened. Outputs are in `results/analysis/historical_known_training`;
checkpoints remain in `../artifacts/s2c/runs/historical_known_training`.
Selected representations still require the same geometry search and a global
selection lock before the one final evaluation. 9/9 remains unverified.

Finalization entrypoint: `scripts/experiments/finalize_historical_known_search.py`.
It requires all nine screening cells and the original 27 geometry units to be
validation-complete. It searches geometry for both screened recipes, chooses
one shared recipe/geometry per dataset/KIR by mean Known utility across three
seeds, and writes all nine locks before opening test. Candidate test maxima are
never computed. Reported wins concern the historical table's numeric references,
not a new matched reproduction of every external baseline. Gate-only Known F1
and accuracy are not cascade metrics. Existing earlier test results remain
historical evidence; this campaign's test access is deferred to finalization.
