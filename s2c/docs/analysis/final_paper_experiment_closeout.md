# Standard Banking77 and official MOGB closeout

Status: complete, 2026-09-15. Base commit: `c2090269f63c9b2ecc2efece7bf8618877f67211`.
The authoritative lightweight bundle is `../../results/final_paper_main/`;
its manifest reports `36/36` complete units. Raw artifacts remain under
`../../../artifacts/s2c/analysis/`.

## Final decision

The final standard-Banking77 Ours is the complete `Gate -> Router -> Expert`
pipeline with a Trainable MiniLM Gate. The Gate uses the same single-centroid
geometry in all three KIR rows: diagonal Mahalanobis distance, a mean-plus-one-
standard-deviation normalized sphere, nearest-sphere acceptance, no score
fusion, and fixed normalized threshold `1.0`. The three KIR rows use the
Known-dev-only recipe winners recorded in the per-cell locks. `banking77_oos`
is historical archive data and is not mixed with this result.

The MOGB matrix is an `official-compatible` reproduction, not a strict
byte-identical reproduction. It runs the pinned public MOGB core and adaptive
granular-ball inference with BERT-base-uncased, on the same canonical rows,
Known labels, evaluator and three seeds as the Ours Banking77 cells.

## Standard Banking77 protocol

| Item | Locked value |
|---|---|
| Source | `../../../textoir/data/banking`; 77 intents; standard 77-intent Banking77 |
| Protocol | `protocol_v2_textoir_v1`; TextOIR benchmark label order plus NumPy `RandomState(seed).choice(replace=False)` |
| Seeds | `13, 42, 87` |
| Known intents | 19 / 38 / 58 for nominal KIR `.25/.50/.75`; actual ratios 0.24675 / 0.49351 / 0.75325 |
| Train/dev | Known intents only; no OOS rows |
| Test | The original 3,080-row test split, with held-out intents mapped to `__oos__` |
| Test composition | `.25`: 760 Known / 2,320 OOS; `.50`: 1,520 / 1,560; `.75`: 2,320 / 760 |
| Selection | Known train/dev only; `pseudo_oos_used=false`, `real_oos_used_for_selection=false` |
| Test selection | `test_used_for_selection=false`; test is read only for final prediction metrics |

The canonical source export, generated views, and MOGB adapter agree on text,
intent, label and sample order. The root manifest also records that earlier
campaigns had observed the test split; therefore this is a locked Known-only
selection result, not an untouched-holdout claim.

## Final Ours configuration

- Gate: `all-MiniLM-L6-v2`-family MiniLM, with the last two Transformer blocks
  trainable. The base recipe is the residual `384 -> 256 -> 384` projection,
  temperature `0.07`, intra/inter weights `0.1/0.1`, margin `0.20`, backbone
  learning rate `2e-5`, projection learning rate `2e-4`, and Known-dev
  checkpoint selection. The locked recipe is `no_projection` at `.25`,
  `temperature014` at `.50`, and `epoch12` at `.75`; their exact epochs and
  flags are in the per-cell lock files.
- Gate geometry: `K=1`, diagonal Mahalanobis, `mean_std_1.0`, normalized
  nearest-sphere rule, no fusion, threshold `1.0`.
- Router: one-domain constant Banking router; no router training is required.
- Expert: `SmolLM-135M` with LoRA rank 16 and alpha 32, trained on Known
  train/dev only; batch size 32, learning rate `2e-4`, weight decay `0.01`,
  maximum length 64, 15-epoch cap and patience 5. Test evaluation is deferred.
- Ours metrics are computed from the same full-pipeline prediction arrays;
  Known macro F1 is computed over the full test set, including OOS-to-Known
  false positives.

## Final standard Banking77 Ours results

Values are percentages; `mean +/- std` uses population standard deviation over
seeds 13/42/87.

| KIR | OOS F1 (seed 13 / 42 / 87; mean +/- std) | Known F1 | Accuracy | Known Recall | False Accept |
|---:|---:|---:|---:|---:|---:|
| .25 | 91.826 / 91.054 / 92.466; **91.78 +/- 0.58** | 76.49 +/- 2.19 | 87.36 +/- 0.98 | 81.71 | 10.10 |
| .50 | 87.172 / 84.998 / 87.014; **86.39 +/- 0.99** | 81.20 +/- 0.89 | 83.77 +/- 1.11 | 80.75 | 9.66 |
| .75 | 69.899 / 72.245 / 69.717; **70.62 +/- 1.15** | 82.13 +/- 0.19 | 78.68 +/- 0.43 | 80.04 | 12.11 |

The main diagnostic is the `.75` class-prevalence shift. Standard Banking77
has 2,320 Known and 760 OOS test rows, so the Ours `.75` cells reject about
436--492 Known rows while accepting 52--120 OOS rows. OOS precision is about
59% and OOS recall is 87.89% on average. The historical paper's `.75`
`banking77_oos` artifact instead had 1,520 Known and 2,560 OOS rows. Its
reported OOS F1 therefore cannot be used as a direct standard-Banking77 target.

Against the reported TextOIR Banking77 references, Ours exceeds the reported
DA-ADB OOS F1 by `+5.21/+4.46/+1.25` pp at `.25/.50/.75`; these references are
not matched-seed reruns and are contextual, not an unconditional SOTA claim.

## Official-compatible MOGB reproduction

Official sources: [MOGB paper](https://ojs.aaai.org/index.php/AAAI/article/view/34630/36785)
and [MOGB repository](https://github.com/Liyanhuaa/MOGB). Local commit:
`5b689e2a03de0d86ec41212825e5db8d7f0e5c02`.

The runner invokes the public `PretrainModelManager.train`, adaptive granular
ball construction and `ModelManager.open_classify`, retaining the released
Euclidean/L1-normalized loss and nearest-ball inference. Only API, shared-data,
legacy optimizer import, eager-attention and memory-offload compatibility
changes were made. The complete matrix has 27/27 manifests, checkpoints,
granular-ball files, predictions and results; all use CUDA, BERT-base-uncased,
Known-dev checkpoint selection and no real OOS training/validation.

| Dataset | KIR | OOS F1 seed 13 / 42 / 87 | Mean +/- std | Known F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| Banking77 | .25 | 90.220 / 90.209 / 91.749 | 90.73 +/- 0.72 | 60.80 +/- 3.98 | 85.01 +/- 1.21 |
| Banking77 | .50 | 79.076 / 77.723 / 78.599 | 78.47 +/- 0.56 | 63.37 +/- 1.21 | 73.05 +/- 0.79 |
| Banking77 | .75 | 52.745 / 54.467 / 55.022 | 54.08 +/- 0.97 | 61.75 +/- 2.12 | 58.85 +/- 1.55 |
| CLINC150 | .25 | 93.717 / 93.536 / 93.364 | 93.54 +/- 0.14 | 62.79 +/- 1.04 | 89.03 +/- 0.24 |
| CLINC150 | .50 | 84.822 / 84.457 / 84.615 | 84.63 +/- 0.15 | 60.75 +/- 0.33 | 78.09 +/- 0.24 |
| CLINC150 | .75 | 71.679 / 71.315 / 71.217 | 71.40 +/- 0.20 | 60.15 +/- 0.44 | 67.20 +/- 0.28 |
| StackOverflow | .25 | 89.877 / 89.791 / 92.797 | 90.82 +/- 1.40 | 66.99 +/- 6.64 | 85.58 +/- 2.30 |
| StackOverflow | .50 | 78.829 / 78.261 / 81.779 | 79.62 +/- 1.54 | 67.51 +/- 4.71 | 75.01 +/- 2.62 |
| StackOverflow | .75 | 58.085 / 57.403 / 57.304 | 57.60 +/- 0.35 | 68.54 +/- 0.79 | 64.36 +/- 0.29 |

The public-paper comparison is a supporting reproduction check, not a matched
ranking. Public MOGB reports StackOverflow `94.42/89.71/75.52` and Banking
`88.29/81.04/71.27` for `.25/.50/.75`. Local minus public is respectively
`-3.60/-10.09/-17.92` pp on StackOverflow and `+2.44/-2.57/-17.19` pp on
Banking77. The large high-KIR gaps, unspecified paper seeds, different paper
dataset counts and released-code/paper discrepancies prevent a strict
reproduction claim, but the official-compatible implementation is complete.
There is no published MOGB CLINC150 row; those three rows are extra
shared-protocol evidence.

## Unified comparison and claim boundary

On the shared standard Banking77 split, Ours minus official-compatible MOGB
OOS-F1 is `+1.06/+7.93/+16.54` pp at `.25/.50/.75`. This is a fair
same-split comparison between the two local artifacts. TextOIR MSP, OpenMax,
DOC, DeepUnk, KNNCL, ADB and DA-ADB values in `comparison.csv` are reported
references with source-specific contracts and are marked
`eligible_for_aligned_main=false`; they must not be pooled into a strict
ranking. The same applies to historical `banking77_oos`, MOGB-MiniLM
components, and any test-oracle or post-hoc best-cell search.

Safe for the paper:

- standard Banking77 Ours values and their Known-only selection provenance;
- the 27-cell official-compatible MOGB baseline, with the compatibility caveat;
- same-split Ours/MOGB comparison and the Known Recall/false-acceptance tradeoff.

Appendix or historical only:

- `banking77_oos` results and the old paper table;
- MOGB public-paper values as contextual references;
- MOGB-MiniLM controlled components, adaptive/test-diagnostic searches and
  reported TextOIR references that were not rerun on the shared split.

## Verification and key paths

- Main manifest: `../../results/final_paper_main/MANIFEST.json` (`36/36`).
- Main tables: `../../results/final_paper_main/per_seed.csv`,
  `summary.csv`, `comparison.csv`, and `mogb_public_comparison.csv`.
- Ours locks and predictions:
  `../../../artifacts/s2c/analysis/banking_textoir_aligned/fixed_k1_mahalanobis_threshold1_known_only/`.
- MOGB raw matrix:
  `../../../artifacts/s2c/analysis/mogb_shared_official_aligned/`.
- Shared dataset audit: `../../results/final_paper_main/dataset_audit.json`.
- Builder: `../../tools/analysis/build_final_paper_main.py`.

The final read-only audit recomputed all 36 cells from saved prediction arrays:
`36/36 PASS`; the nine Banking77 Ours/MOGB test-row pairings were exact. The
targeted regression suite passed `10 tests`.
