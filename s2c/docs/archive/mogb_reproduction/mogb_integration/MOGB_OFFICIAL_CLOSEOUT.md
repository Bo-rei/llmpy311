# MOGB official-logic closeout

## Scope and status

This closeout is separate from the frozen MiniLM-fair MOGB matrices.  It runs
the pinned upstream MOGB training/evaluation flow with the local BERT checkpoint
through the isolated compatibility launcher.  The upstream checkout was not
edited; the launcher only repairs modern PyTorch autograd/device compatibility
and records the repair in the existing source audit.

The result is **converged official-logic compatibility evidence**, not a
strict byte-for-byte reproduction.  The legacy repository has no complete
`utils` package, uses an obsolete BERT stack, regenerates Known labels from its
own TSV contract, and its original autograd path is not executable in the
current environment without the isolated repair.

## Runs

The registered run root is:

```text
../artifacts/s2c/external/mogb_official_converged_v1/
```

Both datasets completed the pre-registered KIR=0.50, seeds
`{13,42,87,100,123}`, five-epoch run.  The upstream result columns mean
`Known=F1-K`, `Open=F1-U`, and `F1-score=F1-All` according to
`third_party/mogb_official/util.py`.

| Dataset | Cells | F1-K mean | F1-U mean | F1-All mean | Accuracy mean | F1-All std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| StackOverflow | 5/5 | 37.5916 | 72.0512 | 40.7243 | 61.4880 | 1.5662 |
| Banking77 | 5/5 | 17.9685 | 69.2843 | 19.2843 | 55.4880 | 2.3308 |

The raw official-format files are retained under each seed's `results/results.csv`.
Every manifest records `strict_official_reproduction=false`, the compatibility
runtime, local BERT path, epoch count, and return code.

## Comparison boundary

The published MOGB table and this run do not share enough protocol fields for a
fair numerical SOTA claim: the paper's original sample construction, Known
intent selection, preprocessing, and training environment differ from
`protocol_v2_textoir_v1`.  The numbers above are therefore an audit result, not
a replacement for the paper table and not a direct comparison with the frozen
MiniLM matrix.

The valid same-protocol comparison remains the completed MiniLM-fair matrix:

* fixed single centroid: overall OOS F1 `0.7808`;
* MOGB-style adaptive balls with official mean radius: overall OOS F1 `0.7339`;
* fixed K=1 beats the adaptive partition in all 45 paired cells;
* `Euclidean + mean+std` is a stronger boundary operating point (`0.7865`)
  but still loses Known coverage relative to the single centroid.

These results separate representation/training from partition and boundary
effects; they must not be labeled official MOGB performance.

## Decision

Official MOGB is now **audited and converged under an isolated compatibility
layer**, but not strict reproduced.  No additional official-BERT sweep is
authorized in the current branch.  Any claim against the paper must use the
paper's own protocol or be explicitly labeled descriptive and non-comparable.

Evidence:

* source audit: `docs/archive/mogb_reproduction/mogb_integration/mogb_official_audit.md`;
* pipeline audit: `docs/archive/mogb_reproduction/mogb_integration/current_pipeline_audit.md`;
* per-run manifests: `../artifacts/s2c/external/mogb_official_converged_v1/`;
* MiniLM-fair closeout: `docs/archive/mogb_reproduction/mogb_integration/MOGB_REPRODUCTION_REPORT.md`.
