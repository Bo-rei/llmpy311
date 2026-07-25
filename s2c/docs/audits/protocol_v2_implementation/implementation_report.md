# protocol_v2 implementation report

## Status

- Base commit: `ea210083b331c489059f275edcc2e0c3241cfba7`
- Dataset version: `protocol_v2_textoir_v1`
- Materialized canonical datasets: `banking77, clinc150, stackoverflow`
- Source revisions: `dffe2b1b848a069a6808f8089b4cb9bd16e2062b`
- Canonical datasets: 3
- Fixed registries: 165
- Materialized views: 165
- Materialized exports: 990
- Completed protocol_v2 runs: 1,686 (36 E1 smoke + 1,650 E2 dense Gate)
- Failed protocol_v2 runs: 0

## Data-admission decision

| Dataset | Provenance decision | Formal admission |
| --- | --- | --- |
| banking77 | accepted_textoir_snapshot | admitted_official |
| clinc150 | accepted_textoir_snapshot | admitted_official |
| stackoverflow | accepted_textoir_snapshot_local_only | admitted_benchmark_local_only |

This report is scoped to the selected dataset version. It preserves frozen and
legacy protocols for audit but does not mix their results into this protocol.

## Materialized data inventory

| Dataset | Samples | Known intents | Native OOS | Local source copy |
| --- | ---: | ---: | ---: | --- |
| banking77 | 13083 | 77 | 0 | `sources/textoir/dffe2b1b848a069a6808f8089b4cb9bd16e2062b/banking77` |
| clinc150 | 23700 | 150 | 1200 | `sources/textoir/dffe2b1b848a069a6808f8089b4cb9bd16e2062b/clinc150` |
| stackoverflow | 20000 | 20 | 0 | `sources/textoir/dffe2b1b848a069a6808f8089b4cb9bd16e2062b/stackoverflow` |

The authoritative local source for this version is the fixed TEXTOIR snapshot
described by the source manifests. TEXTOIR is import-only; no model, view,
export, Gate or Pipeline runtime reads `textoir/data`.

## Completed implementation work

The fixed snapshot is byte-copied into `data/sources`, canonical records preserve original text, labels and
splits, and every method consumes the same registry and fixed views. Gate runs use immutable directories
beneath `artifacts/s2c/runs/protocol_v2_textoir_v1` and keep embedding cache separate from formal evidence.

## Deliberately not claimed

E3--E7 are not experimental evidence until their own manifests exist. Historical v19-v22 artifacts remain
untouched and are not mixed with this protocol. The StackOverflow corpus remains local-only and is excluded
from public Git/result attachments.

## Requirement status

`requirement_matrix.csv` maps the original implementation goals to current evidence. A status of
`complete_local_benchmark` distinguishes local scientific use from public corpus redistribution.

## E2 closeout

The active three-dataset protocol has 36 completed E1 smoke Gate runs and all
1,650 E2 dense Gate runs on banking77, clinc150 and stackoverflow. The E2
integrity audit reports 1,650 planned, 1,650 observed, 0 failed, 0 missing,
0 duplicate and 0 invalid cells. Every run is bound to the frozen base commit,
code patch, resolved configuration, protocol_v2_textoir_v1 canonical manifest,
registry and MiniLM encoder file hashes.

The derived closeout evidence is kept outside Git in
`artifacts/s2c/runs/protocol_v2_textoir_v1/summaries/e2_closeout/`:

- `E2_integrity_report.md`
- `E2_dataset_kir_summary.csv`
- `E2_paired_k_effects.csv`
- `E2_distance_comparison.csv`
- `E2_known_oos_tradeoff.csv`
- `E2_k_selection.csv`
- `E2_k_selection_analysis.md`

E2 uses a fixed Known-only `mean_std` boundary and does not provide a formal
validation-selected K. Test-set oracle K is analysis-only; no E3--E7 experiment
has been started. StackOverflow remains `admitted_benchmark_local_only`: it is
permitted for local training, evaluation and baseline reproduction, but its
corpus must not be tracked in public Git or redistributed by s2c.
