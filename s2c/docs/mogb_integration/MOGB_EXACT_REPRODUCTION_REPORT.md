# MOGB strict single-cell reproduction report

## Scope and isolation

This report covers the registered strict single-cell attempts
`mogb_exact_reproduction_v1` (StackOverflow, KIR=.50, seed=0) and
`mogb_exact_reproduction_banking_v1` (Banking77, KIR=.75, seed=0).  It does
not replace or merge the existing frozen-MiniLM fair matrix, OFAT sweep,
fixed-K ablation, or the historical one-epoch smoke.  The strict runs use the
pinned upstream checkout under `third_party/mogb_official/` without editing
that checkout.

The complete machine-readable evidence is split deliberately:

* lightweight results: `results/mogb_exact_reproduction/` and
  `results/mogb_exact_reproduction_banking/`;
* raw run artifacts and checkpoints:
  `../artifacts/s2c/external/mogb_exact_reproduction_v1/`;
* runner/config: `scripts/experiments/run_mogb_exact_reproduction.py` and
  `configs/baselines/mogb_exact_reproduction_v1.yaml`.

No additional MOGB matrix was started after this closeout.

## Target contracts

The first target was StackOverflow, KIR=0.50, seed=0, using the local
`bert-base-uncased` snapshot.  A second official-format smoke was then run on
Banking77 with the official example contract, KIR=.75, seed=0.  The runner
keeps the official sequence:

1. BERT classification training;
2. train-feature extraction at each epoch;
3. adaptive granular-ball generation;
4. nearest-sub-centroid loss and the second optimizer;
5. epoch-level alternating updates;
6. selected ball centers and mean-radius boundaries;
7. nearest-ball inference;
8. dev-accuracy early stopping and best-checkpoint restoration.

The registered parameters are max sequence length 45, learning rates
`2e-5/1e-4`, train/eval batches `128/64`, maximum 100 epochs, patience 10,
purity thresholds `.90/1.00/.90`, minimum ball sizes `10/5/10`, and update
step 5.  The runner refuses a CPU fallback and records `blocked_no_gpu` if no
CUDA device is available.

## Data contract audit

The local TEXTOIR-compatible snapshot matches the paper's reported shape:

| Check | Observed |
| --- | ---: |
| Classes | 20 |
| Samples per class | 1,000 |
| Train/dev/test | 12,000 / 2,000 / 6,000 |
| Total samples | 20,000 |

The audit also records the ten Known and ten held-out intents, the per-file
SHA256 values, row-level hashes, label order, and duplicate-text counts
(train/dev/test: 4/1/2).  Shape agreement is therefore established, but the
paper does not expose enough information to prove that this is the identical
random sample draw or identical Known-intent list.  The local run must not be
described as a byte-identical reproduction of the published data.

## Seed and compatibility diagnostics

Two pre-registered contracts were executed:

* `official_fixed`: preserves the upstream pre-Data seed 100, followed by the
  Data class reset to seed 0;
* `unified_zero`: explicitly sets all recorded random sources to seed 0.

Both completed 38 epochs, selected epoch 28, produced the same checkpoint
SHA256 (`e43553d9d52a75277726a5d8256b9cca93df43dc1048cd694b9e8e26679137f0`),
and produced byte-identical final metrics.  The compatibility test reports
zero differences for the fixed-centroid formula, CE/sub-centroid synthetic
loss calculation, centroid values, gradient norm, and ball count.  This is a
formula-level diagnostic; it is not a proof that every legacy autograd graph
operation is byte-identical.

## Results and gap to the paper

| Run | Accuracy | F1-All | F1-U | F1-K | Known Recall | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `official_fixed` | 75.1667 | 68.3502 | 79.9676 | 67.1884 | 51.5333 | not reproduced |
| `unified_zero` | 75.1667 | 68.3502 | 79.9676 | 67.1884 | 51.5333 | not reproduced |
| Published reference | 88.6700 | 87.4900 | 89.7100 | 87.2700 | -- | reference only |

The local gaps are `-13.5033`, `-19.1398`, `-9.7424`, and `-20.0816` percentage
points for Accuracy, F1-All, F1-U, and F1-K respectively.  Under the
registered rule (≤3 pp approximate, 3--8 pp partial, >8 pp not reproduced),
the result is `not_reproduced_strict`.

The official metric output additionally shows OOS precision `67.1652`, OOS
recall `98.8`, and 1,449 Known-to-OOS errors.  The full epoch trajectory and
selected-ball statistics are in `training_history.csv` and
`ball_statistics.csv`; no test metric was used for checkpoint or parameter
selection.

## Banking77 KIR=.75 single cell

The Banking77 run uses the local TEXTOIR-compatible snapshot with 77 labels,
13,083 rows and native train/dev/test counts 9,003/1,000/3,080.  It completed
54 epochs, selected epoch 44 by dev accuracy, and used the same official
compatibility layer and empty-child-ball guard as the StackOverflow run.  The
guard only prevents the upstream `exit()` on an empty recursive child; it does
not alter non-empty ball statistics or inference semantics.

| Run | Accuracy | F1-All | F1-U | F1-K | Known Recall | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `official_fixed` | 57.0779 | 59.1627 | 53.1049 | 59.2671 | 43.7069 | not reproduced |
| Published reference* | 80.5800 | 81.5200 | 81.0400 | 81.5300 | -- | descriptive only |

The gaps are -23.5021, -22.3573, -27.9351 and -22.2629 percentage points.
The paper reference is not a fair numerical comparison here: the published
table uses a different KIR setting, while this diagnostic follows the
official `run.sh`-style KIR=.75 example.  It is retained to show that the
single cell does not converge to the published scale, not as a claim about
the exact same split.

## Difference diagnosis

The evidence supports the following bounded conclusions:

* Data shape is consistent with the published StackOverflow protocol, but
  exact sample identity and published Known-intent selection are not
  recoverable from the paper alone.
* Seed-order conflict is not sufficient: the two seed contracts are identical.
* The run converges under the current GPU compatibility layer, so this is not
  an execution failure.
* The selected-ball table contains many small balls (the minimum selected
  size is 10), and the final run has a large Known-recall loss.  Ball selection,
  representation/data-contract differences, and metric/split semantics remain
  plausible causes; none is test-tuned in this run.
* The modern compatibility fixes (device placement, detached feature bank,
  stale-graph lifetime, and batchwise loss handling) are isolated and logged,
  but the synthetic compatibility check cannot identify which of those
  remaining hypotheses explains the published gap.

It would be invalid to tune purity, radius, Known lists, or early stopping
against the published test numbers after observing this gap.  The strict run
is therefore negative reproduction evidence, not an SOTA result.

## BRAK follow-up on MOGB representations

After the completed training run, the same StackOverflow Known split was used
for 18 fixed-K/BRAK summary rows across Frozen MiniLM, initial BERT, and the
trained hierarchical BERT representation.  Frozen MiniLM and initial BERT
selected K=1 for all ten Known intents.  Trained BERT selected K=1 for eight
intents and K=2 for two, but its aggregate OOS F1/F1-All/F1-K/Accuracy were
`0.0000/0.0228/0.0251/0.0282`.  This is a Known-only negative control, not
evidence that adaptive K is ready for expansion.

## Required baseline boundary

DCLOOS remains a separate required end-to-end baseline. Its official source
README points to an external Drive snapshot named `squad.tsv`, while the
upstream loader requests `squad_placeh.tsv`; an isolated runner recorded a
byte-identical rename and started the official BERT contract. The default
cell is `timeout_incomplete`; a distinct reduced-budget cell recovered
prediction-derived metrics (Accuracy `88.6842`, F1-All `90.2629`, OOS F1
`87.0527`) and is recorded as `complete_recovered_intermediate_prediction`.
It is not a strict default/paper reproduction and remains under
`../artifacts/s2c/external/dcloos_official_oos_kir75_seed888_reduced_v2/`,
separate from these MOGB numbers. ADB and DA-ADB remain independent boundary
baselines; their modernized single-cell compatibility results are recorded in
`ADB_DAADB_AUDIT.md`.

## Answers to the requested questions

1. **Data consistency:** reported shape matches, but exact sample draw and
   Known list are not provably identical.
2. **Convergence:** both registered seed contracts converge for 38 epochs and
   restore the same best checkpoint at epoch 28.
3. **Metric gap:** all four paper metrics are more than 8 percentage points
   below the published reference.
4. **Likely source:** no single source is proven; seed order is ruled out as a
   sufficient explanation, while data identity, representation, ball
   selection, and metric/split semantics remain separated hypotheses.
5. **Frozen MiniLM:** it is a component control with a different backbone and
   no hierarchical representation training; it cannot represent official
   MOGB.
6. **BRAK:** it remains conservative on the MOGB representations; only two of
   ten trained-BERT intents select K=2, with poor absolute performance.
7. **Expansion:** the one Banking77 KIR=.75 cell is complete but still fails
   to reproduce the published scale; there is no evidence-based gate for a
   five-seed strict BERT expansion.  No new MOGB/BRAK matrix is authorized by
   this closeout.

## Reproduction classification

`mogb_exact_reproduction_v1` and
`mogb_exact_reproduction_banking_v1 = not_reproduced_strict` under their
respective local contracts.  The Banking result is additionally marked
`kir_mismatch_to_paper_reference` for any direct paper-table comparison.

This classification is intentionally conservative and remains separate from
the historical non-strict compatibility aggregate and the protocol-aligned
frozen-MiniLM fair comparisons.
