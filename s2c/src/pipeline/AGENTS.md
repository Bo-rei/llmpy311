# src/pipeline/ — End-to-End Inference Pipeline

## OVERVIEW
`HiLSAMoEV19Pipeline`: Gate → Router → Expert hierarchical inference with optional semantic gate verification.

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Main pipeline class | `system_pipeline.py` | `HiLSAMoEV19Pipeline.predict_batch()`, `predict_one()` |
| Path configuration | `system_pipeline.py` | `PipelinePaths` dataclass — all component paths |

## CONVENTIONS
- Pipeline entry: `predict_batch(texts) → List[Dict]` or `predict_one(text) → Dict`
- Flow: Gate (OOS filter) → Router (domain) → Expert (fine-grained intent)
- Semantic gate: optional second-stage verification for uncertain samples
- All inference wrapped in `@torch.no_grad()`
- Device defaults to `"cpu"` — never calls `torch.cuda.is_available()` during init

## ANTI-PATTERNS
- **NEVER** call `torch.cuda.is_available()` — use explicit `device` parameter
- **DO NOT** modify `predict_batch()` return schema without updating all downstream consumers
- **DO NOT** load components eagerly — `load()` must be called explicitly after construction

## NOTES
- `gate_mode` controls OOS detection: `"multisphere"` (default) or `"multi_prototype"`
- Semantic gate modes: `"prototype"`, `"llm_verifier"`, `"fusion"`
- Decision policies: `"threshold"` or `"two_stage_verifier"`
- Expert loading is lazy — `_ensure_active_expert()` swaps LoRA adapters on-demand
- `semantic_gate_enabled=False` by default — must be explicitly enabled
