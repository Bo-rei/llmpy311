# Union-Risk Calibrated Subcenter Gates for Known-Only Open Intent Detection

**Method.** Union-Risk Calibrated Subcenter Gate (URCSG)

## Motivation
**Problem framing.** The current s2c Gate is deliberately lightweight: frozen MiniLM embeddings feed per-intent centers, radii, and a minimum normalized distance score. E2 and E3 show that this design has a structural asymmetry. Adding centers can help Banking77, but the same operation can sharply increase StackOverflow false acceptance even when KMeans is stable; therefore cluster stability, effective rank, or a larger K sweep is not a sufficient decision signal.

The missing quantity is the marginal open-space risk created by the union of local regions. A fixed K or lambda is selected globally, while the final detector is a per-intent union; current test-oracle rows are descriptive only, and the existing protocol has no legal validation-unknown split for choosing K. URCSG treats the marginal union event itself as the object to estimate, using rotated known intents as construction-defined pseudo-OOS and retaining the existing detector geometry unchanged.

**Why now.** The need is timely because arxiv:2607.07974v1 makes multi-cluster MiniLM boundaries a direct comparison point, while arxiv:2603.04538v1 shows how hidden calibration and operator mismatch can invalidate benchmark conclusions. The current audit has already frozen the data and provenance contracts, so the next useful step is a small risk-calibration experiment rather than another unregistered architecture or K sweep.

**Why prior work stopped.**
- `arxiv:2607.07974v1` (arXiv 2026-07): arxiv:2607.07974v1 replaced a single MiniLM boundary with fixed multi-cluster local regions for OOS intent detection.
  - _Did not do_: It did not estimate the marginal unknown-space acceptance caused by adding one intent's extra regions under a test-independent protocol.
  - _Structural reason_: The method changes the boundary geometry but does not define a per-intent risk object that can be calibrated before test OOS is observed.
- `arxiv:2602.21252v1` (arXiv 2026-02): arxiv:2602.21252v1 conditioned a detection boundary on context and policy in a different security domain.
  - _Did not do_: It did not test whether a semantic OOS detector's scalar distance score is sufficient after multiple accepted regions are unioned.
  - _Structural reason_: The context-conditioned boundary is not a known-only selection rule for the incremental risk of local-region expansion.
- `arxiv:2603.04538v1` (arXiv 2026-03): arxiv:2603.04538v1 isolated operator and calibration confounds in benchmark comparisons.
  - _Did not do_: It did not construct a per-intent acceptance-union risk estimate that changes an open-intent Gate's final configuration.
  - _Structural reason_: Its diagnostic identifies mismatch but stops before a task-specific feasibility rule and downstream intervention.

**What changes when the gap closes.** The detector gains a legal bridge between Known-only calibration and per-intent multi-center selection: it can keep K=1 for unsafe intents without pretending that a global K is optimal. The paper's claim becomes testable at the level that fails in practice—newly accepted unknown-space events—while the existing embedding, distance, radius, and evaluation contracts remain comparable.

## Method
**Pipeline.** URCSG starts from the unchanged frozen-MiniLM Gate and fits candidate K configurations from $train_{known}.$ It rotates each known intent out of the Known universe, expands only one target intent at a time, and measures newly accepted held-out-intent samples relative to that target's K=1 baseline. A Wilson upper bound and own-intent coverage constraint select the largest feasible K or retain K=1; only after this rule is frozen does one final test pass run. A global geometry-calibrated abstention comparator is reported separately and cannot alter the selector.

### M1
*Contract and detector fitting*

1. **Freeze contracts** (`S1`)
   - Create an immutable run manifest containing $protocol_{version}, dataset_{version}, \mathit{canonical\_manifest\_sha256}, registry_{sha256}, encoder_{revision},$ detector distance, radius method, candidate K set, $epsilon_{cov}, rho_{risk},$ and random seeds. Validate that all source IDs belong to the declared $train_{known}$ or $calibration_{known}$ pools and reject K<1 or empty episode contracts before fitting.
   - _Why:_ Prevents accidental migration or test-driven tuning.
2. **Fit target expansion** (`S2`)
   - For each target intent y, load the cached $train_{known}$ embeddings, fit the existing K=1 centers and radii for every other intent j, then fit K>1 centers and radii only for y using the registered partition, distance, covariance, and radius implementation. Persist center IDs, sample counts, radii, and finite-score checks for both target configurations.

*Existing normalized distance/radius score.*
$$ q_{y,k}(x)=d(z(x),c_{y,k})/r_{y,k} \tag{1} $$

   - _Why:_ Isolates the marginal contribution of one intent's extra regions.

### M2
*Rotated known-only risk measurement*

3. **Rotate held-out intents** (`S3`)
   - Enumerate each calibration intent h as a leave-one-intent-out episode. Remove h from the Known intent universe, retain the same fitted train-only contract on the remaining intents, and mark h calibration rows as pseudo-OOS solely because their held-out identity is construction-defined. Skip episodes with fewer than two remaining intents or zero held-out rows and record the skip reason.

*Target-specific newly accepted pseudo-OOS event.*
$$ I_{y,K,h}(x)=\mathbf{1}[A_{y,K}(x)=1\land A_{y,1}(x)=0] \tag{2} $$

   - _Why:_ Creates a legal proxy for unknown-space exposure without test labels.
4. **Estimate marginal risk** (`S4`)
   - For each target y, candidate K, and eligible episode h, evaluate the target-specific baseline and expanded gates on every held-out row. Store the binary event that the expanded target union accepts a row while the baseline union rejects it; aggregate these events over episodes, compute the Wilson upper confidence bound, and separately compute target-intent calibration coverage and counts.

*Marginal union-risk estimate over eligible held-out intents.*
$$ \Delta U_{y,K}=P_h(I_{y,K,h}(x)=1\mid x\in h) \tag{3} $$

   - _Why:_ Turns union overcoverage into a measurable selection object.

### M3
*Feasibility selection and comparator*

5. **Select per-intent K** (`S5`)
   - For each y and K, mark a candidate feasible only if its own-intent calibration coverage is no lower than the K=1 coverage minus $epsilon_{cov}$ and its Wilson risk upper bound is no greater than $rho_{risk}.$ Select the largest feasible K; if none beyond one is feasible, emit K=1 with a reason code such as $coverage_{loss}, \mathit{risk\_upper\_bound},$ or $ineligible_{episode}.$

*Pre-registered feasibility constraints.*
$$ G_{y,K}\ge G_{y,1}-\epsilon_{cov}\;\land\;UCB_{y,K}\le\rho_{risk} \tag{4} $$

   - _Why:_ Provides a legal fallback rather than a universal K claim.
6. **Fit comparator** (`S6`)
   - Fit a separate global geometry-calibrated abstention comparator from the same $train_{known}$ and $calibration_{known}$ inputs, using the registered frozen embedding and a pre-declared calibration rule. Report its selected global threshold and metrics as a control artifact; it is never passed into the per-intent feasibility selector.
   - _Why:_ Separates target-specific union risk from generic calibration.

### M4
*Frozen evaluation and mechanism ablation*

7. **Freeze and test** (`S7`)
   - Freeze the selected $K_{y}$ map, reason codes, detector parameters, and comparator settings. Run the shared evaluator once on $test_{known},$ held-out OOS, and native OOS pools, emitting F1-All, F1-K, OOS F1, AUROC, AUPR-OOS, Known Recall, false acceptance, false rejection, and per-intent K decisions.
   - _Why:_ Keeps test OOS out of all decisions.
8. **Shuffle negative control** (`S8`)
   - Create a negative-control copy of the episode table in which held-out intent identities are randomly permuted while row counts, embeddings, detector fits, and evaluator calls remain unchanged. Recompute the risk object and selector under the same constants, then compare downstream metrics and $K_{y}$ maps with the unshuffled run.
   - _Why:_ Intervenes on the producer of DeltaU while preserving counts and evaluator.
9. **Compare controls** (`S9`)
   - Assemble a paired comparison table for Single-centroid, fixed K=2, fixed K=4, BRAK, URCSG, the global comparator, and oracle-test-K marked descriptive-only. Join rows by dataset, KIR, seed, registry hash, and representation; report means, dispersion, F1-All, F1-K, OOS F1, Known Recall, FAR, and FRR without using the oracle row for selection.
   - _Why:_ Shows whether any gain is specific to the proposed risk object.

## Reviewer concerns
- **Concern [non_blocking]:** Paper-pointed threat: arxiv:2604.27914v1 $(collision_{hits}).$ Geometry-Calibrated Conformal Abstention is a nearby calibration-family threat because it calibrates geometry-based abstention rather than relying on raw confidence. It does not, from the retrieved collision record, establish a target-specific marginal acceptance event for adding local regions, but a reviewer could reasonably treat the proposed Wilson feasibility rule as a specialized conformal/abstention wrapper unless the candidate measures that marginal event directly and includes a calibrated-abstention comparator.
  - **Response:** The main novelty boundary is stated in $core_{mechanism}$ and the revised method steps: DeltaU is a target-specific marginal event produced by expanding one intent while all other intents remain at K=1, not a global conformal threshold. The candidate also requires a same-contract global geometry-calibrated abstention comparator, directly addressing the Phase 3.2 borderline concern.
  - *Fields changed to address:* `core_mechanism`, `core_mechanism_steps`
- **Concern [non_blocking]:** Un-retrieved mechanism family flagged by the audit (parametric knowledge, not in the retrieved pool): selective classification and conformal risk control; query phrases: class-conditional risk control, leave-class-out abstention calibration — novelty vs this family is UNVERIFIED; run a targeted scoop-check on that vocabulary before investing.
  - **Response:** A scoop-check must determine whether any prior work already constructs a per-intent newly-accepted event for a single-region expansion, rather than merely calibrating a global abstention score. If such a construction exists, the claim must be narrowed to the exact detector union and protocol regime where the prior does not operate.
  - *Fields changed to address:* `core_mechanism`, `core_mechanism_steps`
- **Concern [non_blocking]:** Gap-closure reject-check borderline on gap 1 (C00 (Reframe as a Solvable Object)): Proposing a new attribution definition without an axiomatic or causal argument for why it should count as ground-truth collapses the moment a reviewer asks why this is the real importance.
  - **Response:** The method does not claim that MOGB or fixed K is universally inferior. It uses the frozen s2c detector contract and treats MOGB, BRAK, fixed K, and the global comparator as separate controls, so a positive result must survive same-split paired evaluation.
  - *Fields changed to address:* `core_mechanism`, `core_mechanism_steps`
- **Concern [non_blocking]:** Gap-closure reject-check borderline on gap 2 (C01 (Audit and Pivot an Assumption)): Relocating to a locus already exploited by published work — normalization statistics, activation-similarity losses, semantic concept triggers — collapses the novelty boundary unless the difference is 
  - **Response:** The protocol deliberately forbids test OOS selection and marks oracle-test-K as descriptive only. All selection quantities are built from $train_{known}$ and rotated calibration intents, with manifests, hashes, and a final one-way test pass recorded in the run ledger.
  - *Fields changed to address:* `core_mechanism`, `core_mechanism_steps`

