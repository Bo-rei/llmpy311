# E2 Gate dense sweep closeout

This note records the post-run audit for the active
`protocol_v2_textoir_v1` contract. It does not modify the E2 run directories or
their frozen configuration.

## Coverage

```text
3 datasets × 11 KIR × 5 seeds × 5 K × 2 distances = 1,650
```

The closeout audit found 1,650 observed and completed cells, with zero failed,
missing, duplicate, invalid, or mixed-provenance cells. The three canonical
manifest hashes, E2 configuration hash, E2 plan hash, and code snapshot hash
all match the frozen provenance snapshot.

## Evidence

The detailed derived evidence is intentionally kept with the ignored raw
artifact tree:

`artifacts/s2c/runs/protocol_v2_textoir_v1/summaries/e2_closeout/`

It contains:

- `E2_integrity_report.md`
- `E2_dataset_kir_summary.csv`
- `E2_paired_k_effects.csv`
- `E2_distance_comparison.csv`
- `E2_known_oos_tradeoff.csv`
- `E2_k_selection.csv`
- `E2_k_selection_analysis.md`
- `E2_failed_or_invalid_runs.csv`

Core metrics use a deterministic 10,000-resample paired percentile bootstrap
(SciPy 1.15.3, RNG seed `20260725`). Timing and cluster-size fields are
reported descriptively rather than treated as inferential outcomes.

## Interpretation boundary

E2 uses a fixed Known-only `mean_std` boundary. It does not define a formal
validation-selected K; `oracle_test_best_k` is test-set sensitivity analysis
only. No E3--E7 experiment has been started.

