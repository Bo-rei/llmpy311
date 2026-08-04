# Union-Risk Calibrated Subcenter Gates for Known-Only Open Intent Detection

**Method.** Union-Risk Calibrated Subcenter Gate (URCSG)

## Motivation
Adding several local regions can help a known intent, but it can also make the detector accept more unknown examples. Instead of choosing the number of regions from test performance, URCSG hides one known intent at a time and asks how many extra held-out examples become accepted when only one intent is expanded. It keeps an expansion only when known coverage is preserved and the upper confidence bound on this added acceptance risk stays below a fixed limit.

## Method
### M1
*Contract and detector fitting*

1. Write one manifest with the protocol and data hashes, detector settings, candidate center counts, risk constants, and seeds. Check every sample ID against the declared training or calibration split and stop before fitting if a center count or episode is invalid.
   - _Why:_ The selection must be reproducible and test-independent.
2. Use the existing embedding cache. Keep every non-target intent at one center, and fit the candidate extra centers only for the target intent. Save centers, radii, sample counts, and a check that all scores are finite.

*Existing normalized distance/radius score.*
$$ q_{y,k}(x)=d(z(x),c_{y,k})/r_{y,k} \tag{1} $$

   - _Why:_ This isolates the effect of adding regions to that intent.

### M2
*Rotated known-only risk measurement*

3. Take one calibration intent out at a time. Treat its rows as pseudo-unknown only because the split construction says they are held out, and use the remaining intents for the detector. Record and exclude empty or too-small episodes.

*Target-specific newly accepted pseudo-OOS event.*
$$ I_{y,K,h}(x)=\mathbf{1}[A_{y,K}(x)=1\land A_{y,1}(x)=0] \tag{2} $$

   - _Why:_ The held-out identity is known by construction without using real test OOS.
4. For every held-out row, compare the one-center detector with the target-expanded detector. Count only rows newly accepted by the expansion, average those indicators across eligible episodes, and compute an upper confidence bound plus the target intent's own calibration coverage.

*Marginal union-risk estimate over eligible held-out intents.*
$$ \Delta U_{y,K}=P_h(I_{y,K,h}(x)=1\mid x\in h) \tag{3} $$

   - _Why:_ The quantity directly measures union overcoverage.

### M3
*Feasibility selection and comparator*

5. Apply two fixed checks: preserve nearly the same target-intent coverage and keep the added-acceptance risk below the fixed limit. Keep the largest center count that passes; otherwise record a K=1 fallback and why it happened.

*Pre-registered feasibility constraints.*
$$ G_{y,K}\ge G_{y,1}-\epsilon_{cov}\;\land\;UCB_{y,K}\le\rho_{risk} \tag{4} $$

   - _Why:_ Unsafe intents retain the safe single-center fallback.
6. Run one ordinary global geometry-based abstention baseline with the same data and calibration rules. Report it separately so readers can see whether the target-specific risk object adds value; do not let this baseline choose any intent's center count.
   - _Why:_ A gain must not be explained by generic calibration alone.

### M4
*Frozen evaluation and mechanism ablation*

7. After selection is locked, evaluate once on all declared test pools and save the full metric table and selected center count for every intent. No test result may modify the manifest or selector.
   - _Why:_ Test data only measures the final method.
8. Keep every number of rows and every model calculation the same, but randomly swap which held-out intent an episode is assigned to. Re-run the selector and check that the OOS/coverage advantage disappears if the identity-specific risk signal was causal.
   - _Why:_ The proposed mechanism should disappear when its risk signal is destroyed.
9. Put all methods in one same-split table and include both OOS and Known metrics. Label oracle-test-K as analysis only, and compare methods by matching dataset, split, seed, and representation.
   - _Why:_ The claim is about mechanism-specific improvement, not a leaderboard maximum.

