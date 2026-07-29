# src/legacy/models/ — Historical Model Wrappers & Architectures

## OVERVIEW
PyTorch model wrappers for the HiLSA-MoE pipeline: Router (domain classification), Expert (intent classification), and SVDD Gate (OOS detection) models, all using LoRA adapters.

## STRUCTURE
```
models/
├── transformer.py     # Shared LoRA CausalLM loading + masked pooling
├── expert.py          # SmolLMExpert
├── gate_svdd.py       # SVDDGate (v19.2): SmolLM2-1.7B + LoRA + BiasFreeMLP
├── gate.py            # Gate v18.6: DeepSADLoss, legacy SVDDGate
├── svdd_head.py       # SVDDHead with VarianceRegularization (TASK-202)
├── backbone.py        # SmolLM2Backbone with LoRA support (TASK-201)
└── __init__.py        # Exports: SmolLMExpert, ProjectionHead, SVDDGate, BiasFreeMLP
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Router models | `src/legacy/router/router_model.py` | `QwenRouter`, `SmolLMRouter` — backbone + LoRA + classification head |
| Expert model | `expert.py` | `SmolLMExpert` — backbone + LoRA + projection head + L2 norm |
| Shared encoder | `transformer.py` | Owns the historic `base` state-dict key and masked pooling |
| SVDD Gate (v19.2) | `gate_svdd.py` | `SVDDGate`: SmolLM2-1.7B, LoRA r=64, BiasFreeMLP, ν-SVDD loss |
| SVDD Head | `svdd_head.py` | `SVDDHead` + `VarianceRegularization` — anti-collapse mechanisms |
| Backbone | `backbone.py` | `SmolLM2Backbone` — generative + representation modes |
| Legacy gate | `gate.py` | v18.6 DeepSAD — kept for reference, not active |

## CONVENTIONS
- All models use `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
- LoRA targets: `["q_proj", "v_proj"]`, `bias='none'`, `task_type='FEATURE_EXTRACTION'`
- Mean pooling with attention mask: `sum(hidden * mask) / clamp(mask.sum(), min=1e-9)`
- Router outputs class logits; Expert outputs L2-normalized projected features
- ProjectionHead: 2-layer Linear → ReLU → Linear, configurable bias

## ANTI-PATTERNS
- **NEVER** use `torch.no_grad()` in training forward passes — breaks LoRA gradients
- **DO NOT** mix `gate.py` (v18.6) with `gate_svdd.py` (v19.2) — different architectures
- **DO NOT** change LoRA target modules without verifying checkpoint compatibility

## NOTES
- `SmolLMRouter` and `SmolLMExpert` share identical backbone paths but differ in heads (logits vs. normalized projections)
- `gate.py` is legacy (v18.6) — `gate_svdd.py` is the current v19.2 implementation
- `backbone.py` is from TASK-201, used independently from the main pipeline wrappers
