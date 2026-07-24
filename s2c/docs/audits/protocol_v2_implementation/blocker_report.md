# protocol_v2 scope blocker report

## Decision

The original three-dataset protocol is **not eligible for a formal completion claim**. The current
official version admits only: `banking77, clinc150`. It has 24 completed Gate-only run(s), which
are limited to that admitted scope.

| Dataset | Provenance decision | Admission |
| --- | --- | --- |
| banking77 | reconstructed_from_official | admitted |
| banking77_oos | blocked_unverified | blocked |
| clinc150 | reconstructed_from_official | admitted |
| stackoverflow | blocked_unverified | blocked |

## Blocking condition

StackOverflow has a reproducible public content snapshot, but its raw-source and redistribution-license
chain cannot be independently verified at the record level. Historical `BANKING77-OOS` also lacks a
traceable official OOS-extension source. Neither dataset may be replaced by TEXTOIR, a legacy s2c
prepared copy, a deduplicated StackOverflow variant, or a merged substitute.

## Affected work

- The legacy three-dataset 36-cell E1 smoke cannot be completed with the current evidence.
- The 3,300-cell E2 grid, boundary grid, representation grid, external-method comparison and full
  three-dataset Cascade are not authorised as `protocol_v2_official_v1` claims.
- Existing v19-v22, candidate `protocol_v2`, and historical Cascade outputs remain traceable evidence
  only; they cannot fill a blocked official protocol cell.

## Unblocking evidence

To admit StackOverflow, record one immutable raw source, original file names and SHA256 values, the
20-label mapping and 20,000-row count, a verifiable redistribution license, and a three-way sample/split
comparison against both TEXTOIR and historical s2c inputs. Until then, no training, embedding generation,
MOGB/DCL reproduction or TEXTOIR-fair-comparability claim may use it.
