# Final experiment comparison

| Dataset | Method | KIR | n | OOS F1 | Known F1 | Accuracy |
|---|---|---:|---:|---:|---:|---:|
| banking77 | Ours | 0.25 | 3 | 91.78±0.58 | 76.49±2.19 | 87.36±0.98 |
| banking77 | Ours | 0.5 | 3 | 86.39±0.99 | 81.20±0.89 | 83.77±1.11 |
| banking77 | Ours | 0.75 | 3 | 70.62±1.15 | 82.13±0.19 | 78.68±0.43 |
| banking77 | MOGB-official-compatible | 0.25 | 3 | 90.73±0.72 | 60.80±3.98 | 85.01±1.21 |
| banking77 | MOGB-official-compatible | 0.5 | 3 | 78.47±0.56 | 63.37±1.21 | 73.05±0.79 |
| banking77 | MOGB-official-compatible | 0.75 | 3 | 54.08±0.97 | 61.75±2.12 | 58.85±1.55 |
| clinc150 | MOGB-official-compatible | 0.25 | 3 | 93.54±0.14 | 62.79±1.04 | 89.03±0.24 |
| clinc150 | MOGB-official-compatible | 0.5 | 3 | 84.63±0.15 | 60.75±0.33 | 78.09±0.24 |
| clinc150 | MOGB-official-compatible | 0.75 | 3 | 71.40±0.20 | 60.15±0.44 | 67.20±0.28 |
| stackoverflow | MOGB-official-compatible | 0.25 | 3 | 90.82±1.40 | 66.99±6.64 | 85.58±2.30 |
| stackoverflow | MOGB-official-compatible | 0.5 | 3 | 79.62±1.54 | 67.51±4.71 | 75.01±2.62 |
| stackoverflow | MOGB-official-compatible | 0.75 | 3 | 57.60±0.35 | 68.54±0.79 | 64.36±0.29 |

Reported Banking77 references use different seed/split contracts; see banking_reported_references.csv.
MOGB is eligible only after all 27 official-compatible cells complete with shared predictions.
Ours uses Known train/dev selection and a pre-declared fixed geometry; test deferred.
