# DCLOOS reproduction audit

## Scope

The pinned `liam0949/DCLOOS` checkout is a redirect-only repository.  The
actual source was separately pinned at
`fanolabs/out-of-scope-intent-detection` commit
`588ed54922b427c1f48aa46fe9e7656f0b9f7866`; the redirect commit is
`047f6366e4513066e00a990ccac201b89b174374`.

The source audit and preflight were performed without modifying either
third-party checkout. Source files compile. The initial preflight was blocked
because the local workspace did not contain the external negative corpus; a
follow-up source audit located the official README-linked Drive snapshot and a
new isolated run is now registered separately.

## Supervision contract

| Setting | Pseudo OOS | External open-domain OOS | End-to-end training | Status |
| --- | --- | --- | --- | --- |
| DCLOOS-official | yes (convex combinations) | yes (`squad_placeh.tsv`) | yes | blocked before run |
| DCLOOS-unified | method contract retained | required for faithful comparison | yes | not started |
| s2c/BRAK | no | no | no encoder training | available, Known-only |

The official dataloader also randomly regenerates Known labels, uses the legacy
TSV layout, and contains hard-coded CUDA/identity-comparison paths.  Replacing
`squad_placeh.tsv` with protocol test OOS would change DCLOOS's supervision and
would not be a reproduction.

## External corpus resolution

The official repository README points to a Drive folder whose SQuAD file is
named `squad.tsv`. The pinned upstream loader requests
`squad/squad_placeh.tsv`, so the isolated runner copies the file without any
content change and records the rename assumption. The downloaded file is
50,480,876 bytes, SHA256
`f6bf61866c86d3b11565826c3ca1faa00e31f196e0ad9bfd000ec45575fd426e`, and has
331,711 effective two-column `oos` rows after the upstream loader's malformed
row filtering. The full raw file remains outside Git; only its hash and
provenance are recorded. The code checkout is MIT-licensed, but the Drive
corpus's independent redistribution terms are not asserted by the code
repository; this run is therefore local-only and its raw negative corpus is
not added to public results.

The earlier blocker remains preserved as historical preflight evidence. The
new run root is
`../artifacts/s2c/external/dcloos_official_single_cell_v1/` and is not merged
with protocol_v2 results. It uses the official OOS data layout, the pinned
local BERT snapshot, and an overlay that only repairs removed Transformers
APIs, local model resolution, tokenizer calls, logging, and result
serialization.

Preflight evidence:

```text
../artifacts/s2c/external/dcloos_v1/DCLOOS_PREFLIGHT.json
```

## Decision

The previous `dcloos_external_negative_reaudit_v1` remains
`blocked_missing_external_negative_data`. The newly registered
`dcloos_official_oos_kir75_seed888_v1` used the located corpus, but the BERT
process exceeded the declared three-hour ceiling and was stopped before the
runner could produce final metrics. Its manifest is therefore
`timeout_incomplete`; the intermediate `predictions.npz` is explicitly
excluded from all summaries. It must not be placed in a unified SOTA table
without a completed run and explicit labeling of its external-OOS supervision,
dataset contract, and KIR separately from the Known-only s2c methods.

## Timeout closeout

* Artifact: `../artifacts/s2c/external/dcloos_official_single_cell_v1/`.
* Status: `timeout_incomplete`; no `raw_metrics.json` or final `metrics.json`.
* Runtime: approximately three hours of active BERT training on the isolated
  GPU contract.
* External corpus: `squad.tsv` SHA256
  `f6bf61866c86d3b11565826c3ca1faa00e31f196e0ad9bfd000ec45575fd426e`; the
  runtime `squad_placeh.tsv` copy was byte-identical.
* Decision: do not treat the intermediate predictions as a result and do not
  silently substitute protocol test OOS. A future rerun requires a newly
  registered budget/timeout decision.

## Reduced-budget compatibility recovery

A separately registered reduced-budget cell was run with the same pinned
source, local BERT snapshot, official Drive-linked SQuAD corpus, KIR `0.75`,
and seed `888`, using `max_epochs=100` and `patient=10`.  Training reached a
valid upstream test evaluation and wrote `predictions.npz`, but the overlay
then failed while serializing `raw_metrics.json` because the copied upstream
`main.py` did not import `json`.  The source prediction file is intact and
its logged accuracy/OOS recall agree with an independent recomputation.

The recovery is recorded separately at
`../artifacts/s2c/external/dcloos_official_oos_kir75_seed888_reduced_v2/`:

* `recovery_manifest.json` records the serialization failure and hashes;
* `recovery_metrics.json` is recomputed from the 5,700 prediction rows;
* `run_manifest.json` remains `failed` to preserve the original process exit;
* the result is labeled `complete_recovered_intermediate_prediction`.

Recovered metrics are Accuracy `88.6842`, F1-All `90.2629`, F1-U/OOS F1
`87.0527`, F1-K `90.2916`, OOS precision `88.3752`, OOS recall `85.7692`, and
Known Recall `92.1429` (all percentages).  This is useful end-to-end
compatibility evidence, but it is not a strict default-budget or paper-table
reproduction and must remain separate from the Known-only s2c methods.

The runner now injects the missing `json` import and retains the final-test
fallback for future newly registered runs.  No additional DCLOOS matrix is
authorized by this recovery.
