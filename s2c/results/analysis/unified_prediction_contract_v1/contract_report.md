# Unified prediction contract audit V1

- protocol: `protocol_v2_textoir_v1`
- normalized rows: `2394360`
- method runs: `486`
- aligned method-cell rows: `441`
- non-aligned/external rows: `45`
- invalid rows: `0`

## Contract boundary

Trainable K=1、native MiniLM controls 和 MOGB fair components are read from completed current-protocol artifacts. ADB rows are retained as an external BERT contract and have nullable OOS scores because its runtime artifact exposes final labels rather than the S2C score semantics. DCLOOS and unavailable traditional detectors are listed in `blocked_methods.csv`; no intermediate prediction is promoted to a final metric.

## Audit interpretation

`alignment.csv` compares each method-cell with S2C-Trainable-K1 using deterministic ordered sample-id and label digests. `aligned` means the row sequence and labels match; `external_contract_unaligned` is intentionally not a fair-comparison failure claim, but a boundary against pooling different backbone or supervision contracts.

The compressed row-level artifact is local-only and contains anonymous sample IDs, not raw text or embeddings.
