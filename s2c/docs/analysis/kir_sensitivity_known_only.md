# Standard BANKING77 and KIR sensitivity

Status (2026-09-13): representation screening complete (495 recipe trainings).
The initial BANKING77 Gate-only sweep has 33 final artifacts. Its selection
was incorrectly wired to the older unrestricted Known-utility search rather
than the historical 90% Known-coverage calibration. Corrected selection and
full-pipeline evaluation completed for all 33 BANKING77 units and nine CLINC
anchor units under `coverage_repair/`; nine StackOverflow anchor Gate units
are also complete (its downstream components are still missing).
LaTeX is outside this change.

## Selection regression and repair

Base commit: `c2090269f63c9b2ecc2efece7bf8618877f67211` (dirty worktree retained).
`run_kir_sensitivity_known_only.py` previously chose a checkpoint independently
per seed, then maximized unrestricted Known selective utility over thresholds.
The historical coverage runner instead compares shared recipe/geometry across
seeds, with thresholds calibrated to 90% Known validation coverage. In the
initial BANKING77/.25/seed13 lock, lambda=3 and threshold=2.4 accept 100% of
Known validation and reject no test OOS. This was a method-selection regression,
not evidence that changing Banking datasets alone explains the collapse.

The repaired `geometry` stage reuses the existing two screened recipes and
checkpoints on the new data, the existing coverage search/scoring implementation,
and mean Known utility across seeds. CLINC/BANKING use coverage=.90;
StackOverflow retains its historical fixed K1/Euclidean/mean+1.5std,
normalized-union, threshold=.90 rule. No test value chooses a recipe or threshold.
Historical dataset checkpoints are not transplanted onto standard BANKING77.
The old `locks/` and `final/` remain unchanged; corrected selections and results
are in `../artifacts/s2c/analysis/kir_sensitivity_known_only/coverage_repair/`.
The evaluator also fixes an unconditional downstream-component read in Gate-only
mode and uses the matching coverage scorer, including all fusion definitions.

Completed BANKING77 full-pipeline anchor results (mean±population std, percent):

| KIR | Initial Gate OOS F1 | Corrected pipeline OOS F1 | Known F1 | Accuracy |
|---|---:|---:|---:|---:|
| .25 | 27.43±34.57 | 89.56±0.73 | 75.28±0.34 | 84.55±0.90 |
| .50 | 79.28±2.45 | 80.87±2.11 | 80.40±0.89 | 79.78±1.45 |
| .75 | 70.70±7.81 | 69.64±4.23 | 84.67±0.83 | 80.69±1.51 |

All 33 BANKING77 full-pipeline units are complete. The corrected 11-point
OOS F1 sequence is 93.51±1.24, 88.15±3.83, 89.56±0.73, 84.61±2.67,
84.58±0.85, 80.87±2.11, 75.51±2.82, 71.38±3.78, 69.64±4.23,
61.31±3.87, 53.25±2.72 in ascending KIR order. Source-linked per-seed
metrics, checkpoint paths, geometry, and validation rules are in
`results/analysis/kir_sensitivity_known_only/coverage_repair/per_seed.csv`;
aggregates are in the adjacent `summary.csv`. The existing summary builder
also exports mean±std curves under `figures/kir_sensitivity_known_only/coverage_repair/`.

The .25 collapse is repaired; .75 does not improve. Do not cherry-pick between
initial and corrected test results. These are disclosed repair evaluations on
an already observed test set, not a new untouched holdout or proof of 9/9 SOTA.
The prior manuscript Banking column refers to BANKING77-OOS, so its values are
not matched-standard-BANKING77 comparisons. At .25, Known recall changes from
99.30% to 87.81%, and false acceptance from 78.30% to 15.65%.

The dataset difference is concrete, not just a changed key. Historical
`assets/datasets/s2c/prepared/data/multidataset/v19/banking77_oos/kir75_seed42/MANIFEST.json`
(relative to workspace root) records an intent universe of 50. Historical
Known counts at KIR=.25/.50/.75 are 12/25/38; the standard 77-intent campaign
has 19/38/58. At seed42/.75, historical test has 2,560 OOS out of 4,080 rows,
whereas standard BANKING77 has 760 OOS out of 3,080 rows. Thus the same nominal
KIR denotes different Known class counts and OOS prevalence. The code confirms
the historical builder adds ID-OOS and OOD-OOS sets. This prevents interpreting
the remaining raw score difference as a matched-task implementation regression.

CLINC150 full-pipeline OOS F1 is **95.18±0.48 / 92.13±0.81 / 86.80±1.55**
at .25/.50/.75, matching the current `paper/new_polish.tex` table after rounding.
StackOverflow Gate-only OOS F1 is **94.98±0.93 / 89.97±1.20 / 75.23±1.30**;
do not label those nine artifacts as full pipeline. Its Known F1/Accuracy
require downstream completion. These checks use the same new shared data and
Known-selected checkpoints as the repaired runner.

Validation: four targeted geometry-replay/shared-label/entrypoint tests passed. All 99
train/dev views contain only their manifest's Known intents; BANKING77 has no
exact train/dev text overlap. CLINC retains a few source train/dev repeated texts
(1–3 per affected view); they are not a cause of the BANKING77 regression.

The replacement Banking source is `../textoir/data/banking`. Historical
`banking77_oos` scores are retained but are not results for this campaign.
CLINC150 and StackOverflow retain the historical source data and selection
helpers. The new campaign is not a strict historical H0 reproduction.

Matrix: 3 datasets, 11 KIR values (.10,.20,.25,.30,.40,.50,.60,.70,.75,.80,.90),
seeds 13/42/87, DOC/KNNCL/ADB/DA-ADB/Ours: 495 final method units.

## Implemented and checked

- 99 shared data views under
  `../artifacts/s2c/analysis/kir_sensitivity_known_only/data`.
- Train/dev contain Known rows only; exported TextOIR TSV and Gate JSON have
  identical ordered text/intent pairs, verified across all 99 views.
- Router/Expert training and validation views are materialized.
- TextOIR isolated-runtime Known-list override now changes DataManager,
  whereas the previous adapter only recorded the requested list in metadata.
- Adapter accepts intermediate KIR values; a regression test verifies the
  requested list replaces runtime sampling and updates class counts.
- BANKING77 and CLINC150/StackOverflow representation screening are launched
  as separate CUDA processes. Screening uses the existing 11 recipes and
  Known validation macro F1, with two-recipe expansion to seeds13/87.
- Geometry stage implemented using existing Known-only selective utility.

## Earlier implementation snapshot (superseded by status above)

Finish geometry locks; verify the newly implemented TextOIR deferred-test
state serialization in a native baseline pilot; train downstream components;
implement full-pipeline finalization with a campaign-wide selection barrier;
then aggregate 495 final rows and produce mean/std curves and main-table
candidate rows. No training score is an OOS result. No baseline final run
has completed yet. The DOC pilot is running with the existing easydict shim
after the default Python environment failed its dependency probe. Baseline
training now substitutes Known dev examples for the unused test loader and
serializes the selected method before the test call. Final-state restoration
and campaign-wide finalization are still pending implementation/verification.

Entry: `scripts/experiments/run_kir_sensitivity_known_only.py`.
Stages: `prepare`, `prepare-other`, `train`, `geometry`.
Pass `--datasets banking77 clinc150 stackoverflow` to select the training or
geometry datasets. Default is BANKING77. Existing complete recipe records
are reused; interrupted partial recipe directories need explicit recovery
before restart, following the existing training helper's behavior.

Local manifests/checkpoints and validation histories reside under
`../artifacts/s2c/analysis/kir_sensitivity_known_only`.
The current root manifest describes Banking data preparation, not completion
of the full campaign. Other dataset views have a separate preparation manifest.
