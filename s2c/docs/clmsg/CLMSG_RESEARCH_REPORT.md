# CLMSG Milestone 1--3 research report

## 1. Decision

CLMSG Version C did **not** pass the registered StackOverflow gate. The
Known-only split-conformal layer controlled false rejection, but every required
global, class-conditional, and hybrid local-scale score remained below the
single-centroid OOS F1 on seed 13. Consequently seeds 42/87, local manifold,
label entropy, cross-conformal calibration, and the full three-dataset sweep
were not started.

This is a falsification result, not an incomplete sweep: the execution contract
explicitly allowed later modules only after Version C showed a stable benefit.

## 2. Motivation and relation to MOGB

The frozen-MiniLM MOGB component study completed 270/270 cells before this
stage. Adaptive balls reduced OOS acceptance in some settings but consistently
lost Known coverage; they did not provide a universal replacement for the
single-centroid or fixed-K Gate. CLMSG therefore changed the model class from
center--radius unions to sample-level local support:

\[
\rho_i=d(z_i,\text{kth same-intent neighbour}),\qquad
A_{\mathrm{local}}(z)=\min_i\frac{d(z,z_i)}{\rho_i+\epsilon}.
\]

For Version C, 1,000 Known calibration scores produced

\[
p(z)=\frac{1+\#\{A_j^{\mathrm{cal}}\ge A(z)\}}{n_{\mathrm{cal}}+1},
\qquad p(z)<\alpha\Rightarrow\mathrm{OOS}.
\]

Unlike MOGB, no centroid, ball, radius, recursive split, or representation
training is used. Unlike ordinary KNN, Version B/C normalizes query distance by
the density scale of each support point.

## 3. Protocol and no-OOS declaration

- Protocol: `protocol_v2_textoir_v1`
- Dataset: StackOverflow; KIR=0.50; seed=13
- Proper-train: existing 6,000-row `train_known`
- Calibration: existing 1,000-row Known-only `calibration_known`
- Evaluation: 6,000-row `test_combined`
- Encoder: frozen cached `all-MiniLM-L6-v2`, L2-normalized
- Distance: cosine; `k_neighbors=k_scale=10`
- Fixed alphas: 0.01, 0.025, 0.05, 0.10

The three sample-ID sets are disjoint. Training and calibration contain no OOS
rows. Test data are never used to fit support, estimate local scales, construct
calibration scores, or select alpha. The exact sample-ID, registry, view,
canonical, encoder, and embedding hashes are stored in the run manifest.

## 4. Implemented algorithm variants

1. **KNN-only**: kth-neighbour distance, thresholded by the Known calibration
   order statistic at the predeclared primary alpha 0.05.
2. **Global local-scale**: minimum normalized distance over every support point.
3. **Class-conditional local-scale**: the candidate intent is the ordinary
   nearest support label; normalized search is restricted to that intent.
4. **Hybrid local-scale**: weighted global and class-conditional scores for the
   fixed gammas 0.25/0.50/0.75.
5. **Version C**: each local-scale score receives the same global split-
   conformal calibration at all four fixed alphas.

The candidate intent is never supplied by the test gold label.

## 5. Seed-13 result

All values below come from
`../artifacts/s2c/runs/protocol_v2_textoir_v1/clmsg_v1/support_modes_v1/stackoverflow/kir_0.50/seed_13/metrics.json`
or the read-only MOGB `all_runs.csv` referenced by that run.

| Method | OOS F1 | F1-All | Known Recall | Accuracy | AUROC | AUPR-OOS | False accept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single centroid | 0.6957 | 0.7951 | 0.8820 | 0.7198 | 0.8754 | 0.8218 | 0.4037 |
| Fixed KMeans K=2 | 0.5829 | 0.7313 | 0.8817 | 0.6462 | 0.8267 | 0.7636 | 0.5400 |
| MOGB-MiniLM | 0.7329 | 0.4585 | 0.2897 | 0.6388 | 0.8606 | 0.8016 | 0.0107 |
| MOGB partition + s2c boundary | 0.8025 | 0.6743 | 0.5467 | 0.7583 | 0.8682 | 0.7996 | 0.0260 |
| KNN-only, alpha=0.05 | 0.4520 | 0.6604 | 0.9623 | 0.5938 | **0.8786** | **0.8431** | 0.6970 |
| Local scale, global, natural threshold | 0.0145 | 0.4555 | 0.9990 | 0.3472 | 0.6756 | 0.6723 | 0.9927 |
| Local scale, class-conditional, natural threshold | 0.2899 | 0.6269 | 0.9600 | 0.5280 | 0.6912 | 0.6995 | 0.8237 |
| Version C, global, alpha=0.05 | 0.2402 | 0.4942 | 0.9647 | 0.4095 | 0.6756 | 0.6717 | 0.8587 |
| Version C, class-conditional, alpha=0.05 | 0.3336 | 0.6351 | 0.9493 | 0.5408 | 0.6912 | 0.6983 | 0.7897 |
| Version C, hybrid gamma=0.75, alpha=0.05 | 0.3337 | 0.6340 | 0.9513 | 0.5408 | 0.6928 | 0.7004 | 0.7900 |
| Best descriptive Version C (class, alpha=0.10) | 0.4975 | 0.6678 | 0.8897 | 0.5965 | 0.6912 | 0.6983 | 0.6323 |

The last row is descriptive only; alpha was not selected from test performance.
Even that row trails the single-centroid OOS F1 by 19.82 percentage points and
F1-All by 12.73 points.

## 6. Calibration and Known coverage

Split conformal behaved as intended at the primary alpha:

| Support mode | Target alpha | Empirical Known FR | Coverage error | Known Recall |
| --- | ---: | ---: | ---: | ---: |
| Global | 0.05 | 0.0353 | -0.0147 | 0.9647 |
| Class-conditional | 0.05 | 0.0507 | +0.0007 | 0.9493 |
| Hybrid gamma=0.75 | 0.05 | 0.0487 | -0.0013 | 0.9513 |

Thus the failure is not an inability to hit aggregate Known coverage. At
alpha=0.05, Version C accepts too many OOS examples: false-accept rates remain
0.79--0.86 and OOS recall only 0.14--0.21.

## 7. Score diagnosis

Ordinary KNN is the only promising ranking signal in this pilot: AUROC 0.8786
and AUPR-OOS 0.8431 slightly exceed the single-centroid ranking metrics. Its
fixed alpha=0.05 operating point is too conservative for OOS recall, however.

Local-scale normalization destroys much of that ranking information:

- global local-scale AUPR falls from 0.8431 to 0.6723;
- class-conditional local-scale AUPR is only 0.6995;
- hybrid scores peak at roughly 0.7023 AUPR;
- ordinary nearest-neighbour Known intent accuracy is 90.6%, whereas the
  unrestricted global local-scale support label is only 68.7%.

The mechanism is visible in the support statistics. Mean same-intent local
scale ranges from 0.3619 (`linq`) to 0.6729 (`sharepoint`), with substantial
within-intent ranges. Dividing by the support point's scale preferentially
selects dense points with small denominators, even when their intent is not the
semantically nearest candidate. Class conditioning repairs the label path, but
does not recover OOS ranking. Conformal calibration is monotone in the score,
so it can control coverage but cannot restore lost separability.

## 8. Requested research questions after Milestone 3

1. **Does local nonparametric support beat center--radius?** No at the fixed
   operating points; raw KNN ranking is competitive but its OOS F1 is lower.
2. **Does local-scale normalization improve stability?** No; it is the main
   source of ranking degradation on this cell.
3. **Does conformal calibration control Known false rejection?** Yes at the
   aggregate level.
4. **Does manifold normal residual add OOS gain?** Not tested because Version C
   failed the prerequisite gate.
5. **Does label entropy help StackOverflow?** Not tested for the same reason.
6. **Does CLMSG improve OOS without sacrificing Known Recall?** No.
7. **Does CLMSG beat Single-centroid?** No.
8. **Does it beat Fixed-K?** No on OOS F1 and F1-All at the primary alpha.
9. **Does it beat MOGB-MiniLM?** No on OOS F1; MOGB itself has unusably low
   Known Recall, so the two failures occur at different operating points.
10. **Largest observed contribution?** Split conformal controls Known coverage;
    it does not improve score ordering. Local scale contributes negatively.
11. **Cross-dataset/KIR stability?** Not evaluated because the first gate
    failed.
12. **Should it replace the current paper Gate?** No.

## 9. Failure handling, statistics, and limitations

No significance test is reported from one seed. Seeds 42/87 were deliberately
not run after the seed-13 stop. No UMAP, manifold, entropy, large ablation, or
full Pipeline result is manufactured. The MOGB rows are frozen-MiniLM component
references, not an official BERT reproduction.

The current result does not prove that every KNN or conformal OOS method fails.
It falsifies the specific support-point local-scale formulation and fixed-alpha
Version C under the registered pilot. Raw KNN's strong AUROC/AUPR indicates
that a future project could study coverage--OOS Pareto calibration, but that is
a different registered question and is not authorization to tune alpha on test.

## 10. Stage decision

`stop_after_seed13`. Retain the implementation and artifacts as negative
evidence; do not implement local manifold or entropy, do not run seeds 42/87,
and do not start the full CLMSG sweep. The next research action is paper/claim
closeout using the now-complete fixed-centroid, MOGB, and CLMSG evidence, not a
new unregistered Gate expansion.

## 11. Verification

The dedicated verifier confirmed 26 authorized outputs, 156,000 prediction
rows, 1,000 calibration scores, disjoint proper-train/calibration/test sample
IDs, and no test-based selection. The full project suites passed with 258 unit,
8 integration, and 3 smoke tests. Ruff, compileall, research-state,
data-tracking, development-log, experiment-registry, public-result hash, and
Git whitespace checks also passed. The worktree remains deliberately dirty;
no commit or push was performed.
