# DCLOOS reproduction audit

## Scope

The pinned `liam0949/DCLOOS` checkout is a redirect-only repository.  The
actual source was separately pinned at
`fanolabs/out-of-scope-intent-detection` commit
`588ed54922b427c1f48aa46fe9e7656f0b9f7866`; the redirect commit is
`047f6366e4513066e00a990ccac201b89b174374`.

The source audit and preflight were performed without modifying either
third-party checkout.  Source files compile, but no faithful training smoke was
started because the official negative corpus is unavailable locally.

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

## Blocker

The local TEXTOIR-compatible snapshot contains train/dev/test files but no
official open-domain `squad/squad_placeh.tsv` (nor equivalent StackOverflow or
Banking negative file).  A local BERT checkpoint is present, so missing model
weights are not the blocker.  Without the negative corpus and an explicit
legacy-to-protocol data adapter, no official or unified DCLOOS metric is
reported.

Preflight evidence:

```text
../artifacts/s2c/external/dcloos_v1/DCLOOS_PREFLIGHT.json
```

## Decision

DCLOOS remains `blocked_missing_official_open_domain_oos`.  It must not be
placed in a numerical SOTA table as zero or as a fabricated result.  If the
required external corpus is later supplied with a license and a fixed hash,
register a new run root and compare it separately from Known-only s2c methods.
