# Standard Banking77 final comparison

This namespace is a manifest-backed standard `banking77` source for the current Ours result. It excludes `banking77_oos`.
All Ours configurations were selected from Known train/dev only; test was read only for the final predictions stored here.

| KIR | Ours OOS F1 | Strongest reported reference | Delta (pp) |
|---:|---:|---|---:|
| 0.25 | 91.78±0.58 | DA-ADB (86.57) | +5.21 |
| 0.50 | 86.39±0.99 | DA-ADB (81.93) | +4.46 |
| 0.75 | 70.62±1.15 | DA-ADB (69.37) | +1.25 |

Reference rows are reported TextOIR values and are not shared-seed reruns in this namespace; numerical superiority is therefore not a substitute for a matched direct comparison.
