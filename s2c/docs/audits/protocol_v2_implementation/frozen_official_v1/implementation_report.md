# protocol_v2 implementation report

## Status

- Base commit: `fcd9df5249a7e5388080277795c81afa49ed6f7d`
- Dataset version: `protocol_v2_official_v1`
- Materialized canonical datasets: `banking77, clinc150`
- Source revisions: `57ec275d8078af65b7731c2a98be812d844a6d6b, 828f8093932c8fe6ca7936c3d2e52903b1c523de`
- Canonical datasets: 2
- Fixed registries: 286
- Materialized views: 8
- Materialized exports: 32
- Completed protocol_v2 runs: 24
- Failed protocol_v2 runs: 0

## Data-admission decision

| Dataset | Provenance decision | Formal admission |
| --- | --- | --- |
| banking77 | reconstructed_from_official | admitted |
| clinc150 | reconstructed_from_official | admitted |

This report is scoped to the selected dataset version.  It does not upgrade a
blocked dataset, a historical candidate snapshot, or a legacy experiment into
official evidence.

## Materialized data inventory

| Dataset | Samples | Known intents | Native OOS | Local source copy |
| --- | ---: | ---: | ---: | --- |
| banking77 | 13083 | 77 | 0 | `sources/official/polyai-banking77-57ec275d8078af65b7731c2a98be812d844a6d6b/banking77` |
| clinc150 | 23700 | 150 | 1200 | `sources/official/clinc-oos-eval-828f8093932c8fe6ca7936c3d2e52903b1c523de/clinc150` |

The authoritative raw source for this version is the source manifest above;
TEXTOIR is retained only as a three-way audit and export-format reference, not
as a runtime dependency.

## Completed implementation work

The approved raw source is byte-copied into `data/sources`, canonical records preserve original text and
splits, and each experimental method consumes the same registry and fixed views. `textoir/data` is not a
runtime dependency. Gate runs use immutable directories beneath `artifacts/s2c/runs/protocol_v2_official_v1`
and keep embedding cache separate from formal evidence.

## Deliberately not claimed

Declared boundary, representation, external-baseline and full-pipeline matrices are not experimental evidence
until their run manifests exist. Historical v19-v22 artifacts remain untouched and are not mixed with this
protocol. The StackOverflow corpus remains local-only because its redistribution licence is not verified.

## Requirement status

`requirement_matrix.csv` maps the original implementation goals to current evidence. A status of
`complete_for_admitted_scope` applies only to the two officially admitted datasets; it never upgrades the
blocked three-dataset protocol into a completed claim.

## Remaining gate

The completed Gate evidence is limited to 24 run(s) on admitted dataset(s): banking77, clinc150. The legacy three-dataset E1 is intentionally not completed because blocked StackOverflow (blocked) cannot enter this official version. Before a new model experiment is accepted, its own dataset admission, materialized views/exports, runtime-independence check and targeted tests must pass. See `experiment_plan.csv`, `experiment_coverage.csv` and `failed_runs.csv` for the
current state rather than inferring completion from configuration files.
