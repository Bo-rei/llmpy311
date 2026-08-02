# MOGB 接入审计与分阶段执行计划

## 项目结构与关键入口

当前活动代码根是 `s2c/`，不是 `src/s2c/` 的嵌套项目。MOGB 相关入口如下：

```text
third_party/mogb_official/                  pinned upstream checkout (read-only)
third_party/mogb_compat/                    isolated compatibility reference
src/protocol_v2/experiments/mogb/            frozen-MiniLM component adapter
scripts/experiments/run_mogb_fair.py        one fair component cell
scripts/experiments/run_mogb_sweep.py       fair sweep/resume entrypoint
scripts/experiments/run_mogb_exact_reproduction.py
                                             strict official single-cell runner
scripts/experiments/run_brak_mogb_representations.py
                                             BRAK representation-transfer runner
configs/baselines/mogb_exact_reproduction_v1.yaml
docs/mogb_integration/                       audits and closeouts
results/mogb/                                 existing lightweight fair exports
results/mogb_exact_reproduction/              strict/BRAK lightweight exports
../artifacts/s2c/external/                   checkpoints and full manifests
```

The upstream checkout is never imported into the active package and is not
edited in place. Any runtime compatibility patch is materialized in an
isolated run directory and recorded in the run manifest.

## Data-flow comparison

```text
s2c canonical + registry + views
    -> immutable MiniLM cache -> fixed K / random / MOGB partition
    -> shared s2c evaluator

official MOGB source snapshot + legacy TSV contract
    -> BERT CE and projection -> epoch feature bank
    -> GBNR adaptive balls + nearest sub-centroid loss
    -> mean-radius nearest-ball evaluation
```

The first path answers a fair component question. The second path answers
whether the pinned official training logic can be executed and compared with
the published reference. Their metrics are never silently combined.

## Reusable modules

* `protocol_v2` canonical/registry/view loaders and hashes;
* existing MiniLM cache and shared OOS evaluator;
* `src/protocol_v2/experiments/mogb` partition and ball diagnostics;
* pinned MOGB `cluster3.py`, loss modules and result formatting through the
  compatibility runner;
* Known-only BRAK selector for representation transfer diagnostics;
* `tools.compat.textoir` isolated overlay and data-contract audit helpers.

## Files changed or added in this stage

* `configs/baselines/mogb_exact_reproduction_v1.yaml`;
* `scripts/experiments/run_mogb_exact_reproduction.py`;
* `scripts/experiments/run_brak_mogb_representations.py`;
* `scripts/experiments/build_final_baseline_summary.py`;
* `docs/mogb_integration/MOGB_REPRODUCTION_REPORT.md`;
* `docs/mogb_integration/ADB_DAADB_AUDIT.md`;
* research status, ledger, decision log, development log and final baseline
  summary.

Large checkpoints, BERT features, raw predictions and third-party source
remain outside Git-tracked lightweight results.

## Main risks and controls

| Risk | Control |
| --- | --- |
| Official source lacks complete legacy dependencies | Pin source hash, use an isolated compatibility overlay, and label the result non-strict when the original contract is not recoverable. |
| Official data/split differs from s2c | Audit 20,000-row/20-label shape, source tree hash, Known list and train/dev/test counts before training. |
| Seed order changes the result | Run both `official_fixed` and `unified_zero`, record every seed source, and compare checkpoint/metric hashes. |
| Mean-radius balls reject Known samples | Report Known Recall and F1-All alongside OOS F1; do not select a paper claim from OOS F1 alone. |
| BRAK silently uses test labels | Keep selection on train and Known calibration only; write `test_used_for_selection=false`. |
| DCLOOS is mistaken for a boundary baseline | Keep its pseudo-OOS plus external-OOS contract and blocker separate from ADB/DA-ADB. |

## Execution plan and stop rules

1. Audit source, pipeline and data contracts.
2. Complete strict StackOverflow single-cell MOGB under both seed contracts.
3. Run BRAK only on Frozen, initial-BERT and trained-BERT representations.
4. Audit ADB/DA-ADB runnability without fabricating metrics.
5. Keep DCLOOS independently blocked until its fixed external negative corpus
   and license are available.

Steps 2 and 3 are complete. The strict result is `not_reproduced`, so no
official-BERT sweep is authorized. The BRAK result is a negative control, so
no adaptive-K expansion is authorized. The next permitted change is a newly
registered, isolated baseline run after the relevant dependency or DCLOOS
data blocker is resolved.
