# CLINC Known-only validation calibration

Base commit: `d0b10e267a3ca41c2130b804e26900c1e5c53077`.

Historical H1 CLINC150, KIR .25/.50/.75, seeds 13/42/87; existing
Known-validation-selected Trainable MiniLM checkpoints, K=1, diagonal
Mahalanobis, lambda=1. This is Gate-only evaluation.

Primary rule maximizes Known-validation intent macro F1, with the smallest
threshold breaking ties. Secondary rule fixes 95% Known validation coverage.
Both rules are declared before this run's test access and reported separately.
The nine cells are all locked before opening any test file. OOS validation
rows are excluded. Existing test results have already informed the research
direction; these are benchmark follow-up results, not an independent holdout.

Outputs: `results/analysis/historical_known_validation/selection_lock.json`,
`per_seed.csv`, `summary.csv`, and `MANIFEST.json`.

Status: complete, nine CUDA calibration/evaluation cells. No downstream inference was performed.

OOS F1 (three-seed mean): Known macro F1 selection gives 56.46/73.46/74.20
at KIR .25/.50/.75; 95% coverage gives 89.14/80.14/77.65. Both rules fail
to improve the original fixed-threshold result.

The subsequent six CUDA training cells in
`results/analysis/historical_known_representation/MANIFEST.json` also completed.
Known-validation checkpoint selection gives OOS F1 **89.64±0.77** at KIR .50
and **81.37±0.54** at .75 (population standard deviation). The external table
references are 90.10 and 86.00, leaving gaps of 0.46 and 4.63 percentage points.
Known Recall is 72.37/72.95; OOS false acceptance is 3.24/2.27 percent.
These are Gate-only results; full-pipeline confirmation has not been run.
The nine-of-nine objective remains unmet.
