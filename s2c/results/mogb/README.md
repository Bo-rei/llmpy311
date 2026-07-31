# MOGB baseline evidence

This directory contains lightweight, protocol-aligned MOGB fair summaries.
The full run directories, granular-ball trees and sample-level predictions stay
under `../artifacts/s2c/runs/protocol_v2_textoir_v1/mogb_baseline_v1/` and are
not public Git results.

`fair_matrix.csv` compares fixed and adaptive boundary components on the same
`protocol_v2_textoir_v1` registry, split and frozen MiniLM cache. It contains
270 cells: three datasets, three KIR values, five formal seeds and six
protocol-aligned methods. It is a MiniLM component comparison, not a strict
official BERT/TextOIR reproduction.

`official_preflight.json` records why the pinned legacy BERT/TextOIR source is
not yet a strict official reproduction.  `mogb_minilm` and the two component
hybrids must not be labeled as the authors' full MOGB method.
