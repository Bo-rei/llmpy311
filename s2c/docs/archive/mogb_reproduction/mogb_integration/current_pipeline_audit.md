# Current protocol pipeline audit for MOGB integration

Audit base: `294d6f2` (2026-07-29). Existing E0--E3 and MiniLM pilot artifacts
were read-only inputs and were not modified.

| Component | Current implementation | Reuse decision |
| --- | --- | --- |
| Canonical data | `protocol_v2.data` with `protocol_v2_textoir_v1` | Reuse |
| Known/OOS split | Fixed registry and materialized `gate/train.json`, `val.json`, `test.json` | Reuse exactly |
| MiniLM | local `assets/models/all-MiniLM-L6-v2`; canonical embedding cache | Reuse cache; no implicit encoding |
| Normalization | `MultiSphereOOSDetector._normalize_embeddings`, L2 per row | Reuse for fair MiniLM modes |
| Fixed partition | `MultiSphereOOSDetector(center_mode=class_centroid_mixture)`; K is per intent | Keep as control |
| Random control | `experiments.partitions.build_partition(..., random_balanced)` | Keep as control |
| Ours boundary | `mean_std`, `lambda=1.0`, Euclidean or diagonal Mahalanobis | Keep unchanged |
| Ours score | historical `nearest_sphere`, score `distance/radius`, threshold 1 | Keep unchanged |
| Metrics | `compute_binary_oos_metrics`, OOS is positive | Reuse; add F1-All/F1-K/F1-U for baseline report |
| Provenance | registry, canonical/view/export manifest hashes and embedding cache hashes | Record in each MOGB run |

The fair runner is intentionally separate from `experiments.runner.run_gate`:
it must not expand the frozen E2 representation matrix or alter its run schema.
All MOGB output is written below
`artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_baseline_v1/`.

## Data flow

```text
protocol_v2 canonical + registry
    -> materialized gate views
    -> cached all-MiniLM vectors (validated by sample IDs and bytes)
    -> either fixed/KMeans controls or adaptive granular balls
    -> one shared open-set evaluator
```

No runtime path under `textoir/data` is accepted.  The official MOGB BERT path
is recorded separately because it has a different data format and training
contract.
