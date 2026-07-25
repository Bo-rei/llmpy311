# protocol_v2 active-protocol constraint report

## Decision

`banking77, clinc150, stackoverflow` is the active fixed TEXTOIR-compatible local benchmark scope. It has
36/36 completed E1 Gate-only smoke run(s) and 1,650/1,650 E2 dense Gate run(s).

| Dataset | Provenance decision | Admission |
| --- | --- | --- |
| banking77 | accepted_textoir_snapshot | admitted_official |
| banking77_oos | legacy_only | legacy_only |
| clinc150 | accepted_textoir_snapshot | admitted_official |
| stackoverflow | accepted_textoir_snapshot_local_only | admitted_benchmark_local_only |

## Local-only boundary

StackOverflow is a fixed 20,000-title, 20-label TEXTOIR-compatible snapshot for **local** scientific
experiments. Its provenance does not establish a per-row redistribution licence. Consequently s2c must
not track its complete text in Git, repackage it in an appendix, call it an official Stack Overflow
classification release, or claim complete per-row attribution. These limits do not block canonical
construction, embedding generation, Gate/Pipeline experiments, or external baseline reproduction.

## Affected work

- `protocol_v2_official_v1` is frozen for audit and may not be mixed with this active protocol.
- Legacy `protocol_v2` remains rejected and may not be revived as a formal result source.
- E2 is complete and has a dedicated integrity/paired-effects closeout under
  `artifacts/s2c/runs/protocol_v2_textoir_v1/summaries/e2_closeout/`. E3--E7
  remain deliberately unstarted until that closeout is reviewed.

## Unblocking evidence

If public redistribution becomes necessary, a separate source/licence review is required. It is not a
precondition for the present local benchmark protocol.
