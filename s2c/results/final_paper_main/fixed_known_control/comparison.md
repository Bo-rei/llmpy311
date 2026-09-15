# Final experiment comparison

| Dataset | Method | KIR | n | OOS F1 | Known F1 | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| banking77 | Ours | 0.25 | 3 | 90.70±0.87 | 75.50±1.34 | 85.81±1.16 |
| banking77 | Ours | 0.5 | 3 | 82.50±0.15 | 80.58±0.73 | 80.77±0.49 |
| banking77 | Ours | 0.75 | 3 | 73.24±3.42 | 84.53±0.77 | 81.35±1.18 |
| banking77 | MOGB-official-compatible | 0.25 | 3 | 91.01±0.38 | 62.98±1.19 | 85.54±0.61 |
| banking77 | MOGB-official-compatible | 0.5 | 0 | unavailable | unavailable | unavailable |
| banking77 | MOGB-official-compatible | 0.75 | 0 | unavailable | unavailable | unavailable |
| clinc150 | MOGB-official-compatible | 0.25 | 0 | unavailable | unavailable | unavailable |
| clinc150 | MOGB-official-compatible | 0.5 | 0 | unavailable | unavailable | unavailable |
| clinc150 | MOGB-official-compatible | 0.75 | 0 | unavailable | unavailable | unavailable |
| stackoverflow | MOGB-official-compatible | 0.25 | 0 | unavailable | unavailable | unavailable |
| stackoverflow | MOGB-official-compatible | 0.5 | 0 | unavailable | unavailable | unavailable |
| stackoverflow | MOGB-official-compatible | 0.75 | 0 | unavailable | unavailable | unavailable |

Reported Banking77 references use different seed/split contracts; see banking_reported_references.csv.
MOGB is eligible only after all 27 official-compatible cells complete with shared predictions.
Ours uses a fixed geometry and pure Known-dev calibration; previously observed test is disclosed.
