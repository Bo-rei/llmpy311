# src/protocol_v2/ — Active Protocol Package

## Scope

This is the only active Python package for `protocol_v2_textoir_v1`. Keep all
current data contracts, experiment runners, evaluation metrics, active Gate
logic, path resolution, and provenance tracking here.

## Import boundary

- Use `protocol_v2.*` for active code.
- A compatibility call into `legacy.*` must be explicit and documented.
- Never import `src.*` or the historical `s2c.*` namespace.
- Do not put a script, checkpoint, embedding, or generated result under this
  package.

## Subpackages

| Package | Responsibility |
| --- | --- |
| `data` | canonical data, registries, views, and fixed-split exports |
| `evaluation` | metric definitions and score-direction contracts |
| `experiments` | active plans, runners, summaries, and diagnostics |
| `gate` | active Gate implementation and boundary semantics |
| `runtime` | active workspace and artifact paths |
| `tracking` | manifests, provenance, and atomic run writes |

Changes to the active Gate contract require a regression test and must not
modify frozen historical artifacts.
