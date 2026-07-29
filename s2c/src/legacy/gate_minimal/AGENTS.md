# src/legacy/gate_minimal/ — Strict Deep SVDD Baseline

## OVERVIEW
Minimal Deep SVDD implementation strictly aligned with Ruff et al., "Deep One-Class Classification" (ICML 2018). No engineering optimizations allowed.

## STRUCTURE
```
gate_minimal/
├── model.py              # MinimalSVDDGate: pure MLP, bias-free, ReLU/LeakyReLU
├── model_multicenter.py  # Multi-center variant (when needed)
├── loss.py               # Soft-boundary Deep SVDD loss (Eq. 3)
└── loss_multicenter.py   # Multi-center loss variant
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| SVDD gate model | `model.py` | `MinimalSVDDGate`: input → 1024 → 256 → svdd_dim, all bias=False |
| SVDD loss | `loss.py` | `SVDDLoss`: R² + hinge + L2 reg, ν controls outlier fraction |
| Multi-center model | `model_multicenter.py` | Multiple centers for complex boundaries |
| Multi-center loss | `loss_multicenter.py` | Loss for multi-center variant |

## CONVENTIONS
- Center `c` is always a `register_buffer` (never a parameter, never participates in gradients)
- All `Linear` layers must use `bias=False` — enforced by assertions
- No BatchNorm, LayerNorm, or Dropout allowed
- Only ReLU or LeakyReLU (unbounded activations)
- Radius R computed from (1-ν) quantile, detached from gradient computation

## ANTI-PATTERNS
- **NEVER** add engineering optimizations (projection heads, pre-trained backbones) — this is a strict ICML 2018 baseline
- **NEVER** use bias in Linear layers
- **NEVER** add normalization layers
- **DO NOT** modify center during training — only via `update_center()` during initialization

## NOTES
- `input_dim=384` matches all-MiniLM-L6-v2 embeddings (pre-computed, no backbone in forward pass)
- Center must be initialized from Known data before training (via `init_center.py` or similar)
- Slack ratio should converge to ≈ ν (±2%)
