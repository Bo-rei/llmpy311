# Standard Banking77 final comparison

This namespace is the primary standard `banking77` source for the current Ours result. It excludes `banking77_oos`.
All Ours configurations were selected from Known train/dev only; test was read only for the final predictions stored here.

| KIR | Ours OOS F1 | Strongest reported reference | Delta (pp) |
|---:|---:|---|---:|
| 0.25 | 87.63±2.12 | DA-ADB (86.57) | +1.06 |
| 0.50 | 81.87±2.48 | DA-ADB (81.93) | -0.06 |
| 0.75 | 70.10±0.75 | DA-ADB (69.37) | +0.73 |

Reference rows are reported TextOIR values and are not shared-seed reruns in this namespace; numerical superiority is therefore not a substitute for a matched direct comparison.
