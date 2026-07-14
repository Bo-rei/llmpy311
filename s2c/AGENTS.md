# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-04
**Project:** HiLSA-MoE v19 — Hierarchical LLM-based Mixture-of-Experts Intent Classifier

## OVERVIEW
Hierarchical intent classification pipeline: Gate (OOS detection) → Router (domain classification) → Expert (fine-grained intent). Built on SmolLM-135M/1.7B + LoRA, PyTorch, sentence-transformers. Conda env `bo`, Python 3.11.

## STRUCTURE
```
./
├── src/              # Core library code
│   ├── gate/         # OOS detection gates (multi-sphere, multi-prototype, LLM verifier)
│   ├── gate_minimal/ # Strict Deep SVDD baseline (ICML 2018)
│   ├── models/       # Shared encoder, Expert, and SVDD model components
│   ├── pipeline/     # End-to-end inference pipeline
│   ├── inference/    # ExpertManager (memory-efficient LoRA switching)
│   └── utils/        # Data loader, model factory
├── tools/            # Analysis & eval scripts (versioned v19)
├── scripts/          # Data build/rebuild scripts
├── configs/          # Hydra YAML configs
├── tests/            # pytest tests
├── eval/             # Standalone OOS eval
├── docs/             # Chinese-language project docs (entry: docs/README.md)
├── outputs/          # Experiment artifacts, checkpoints, reports
├── data/             # Datasets
├── archive/          # Historical cleanup outputs
└── all-MiniLM-L6-v2/ # Local sentence-transformer model
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Run end-to-end inference | `src/pipeline/system_pipeline.py` | `HiLSAMoEV19Pipeline.predict_batch()` |
| OOS gate logic | `src/gate/multi_prototype_gate.py` | K-means prototypes, cosine similarity |
| OOS sphere detection | `src/gate/multi_sphere_oos_detector.py` | Multi-sphere SVDD |
| LLM semantic verification | `src/gate/llm_semantic_verifier.py` | Router-based semantic check |
| Router models | `src/router/router_model.py` | QwenRouter, SmolLMRouter |
| Expert model | `src/models/expert.py` | SmolLMExpert |
| Shared encoder | `src/models/transformer.py` | LoRA CausalLM loading and masked pooling |
| SVDD gate model | `src/models/gate_svdd.py` | SVDDGate, BiasFreeMLP |
| Minimal SVDD baseline | `src/gate_minimal/` | Strict ICML 2018, no engineering optimizations |
| Expert LoRA switching | `src/inference/expert_manager.py` | Single base + 10 LoRA adapters |
| Gate training config | `configs/v19_gate.yaml` | SmolLM2-1.7B backbone |
| Main training config | `configs/config.yaml` | Hydra, hilsa-llm-v4 experiment |
| Run analysis/eval | `tools/analysis/`, `tools/eval/` | v19-versioned scripts |
| Project docs (CN) | `docs/README.md` | Entry point for Chinese docs |

## CONVENTIONS
- **Versioning**: All experiment scripts suffixed `_v19` (e.g., `run_pipeline_ablation_matrix_paper_v19.py`)
- **Hydra**: Config composition via `configs/config.yaml` with defaults list
- **Import style**: `from src.module import Class` — repo root on sys.path
- **Model wrappers**: LoRA via `peft`, backbone frozen, only LoRA + head trainable
- **Data format**: JSON arrays of `{text, intent, domain, label}` objects
- **Gate modes**: `multisphere` (default) or `multi_prototype` — controlled by `gate_mode` param

## ANTI-PATTERNS (THIS PROJECT)
- **NEVER** call `torch.cuda.is_available()` during init — can trigger native runtime aborts in some envs
- **NEVER** use `torch.no_grad()` in training loops — breaks LoRA gradient flow (already fixed in `SmolLMExpert`)
- **DO NOT** modify `gate_minimal/` with engineering optimizations — must stay strict to ICML 2018 paper
- **DO NOT** introduce new config files parallel to `configs/` — use Hydra composition
- **DO NOT** write new docs as parallel `.md` files — update existing `docs/` files only

## UNIQUE STYLES
- Gate uses two-threshold uncertainty region (`tau_low`, `tau_high`) for accept/uncertain/reject
- Semantic gate supports 3 modes: `prototype`, `llm_verifier`, `fusion` (weighted alpha/beta)
- ExpertManager uses single shared base model with dynamic LoRA adapter switching (memory-efficient)
- `gate_minimal/` is a deliberate minimal baseline — no projection heads, no multicenter unless in `*_multicenter.py`

## COMMANDS
```bash
# Activate env
conda activate bo

# Run tests
pytest tests/

# Run gate training (example)
python -m src.gate_minimal.model  # adjust per script

# Run system pipeline eval
python tools/eval/eval_system_pipeline_v19.py

# Run ablation matrix
python tools/analysis/run_pipeline_ablation_matrix_paper_v19.py
```

## NOTES
- `src/router/` is the only Router implementation location; do not re-export Router classes from `src/models/`.
- `environment.yml` has pinned versions — do not casually update
- Chinese docs in `docs/` are the authoritative project documentation
- `outputs/` contains all experiment artifacts — do not delete without archiving
- `archive/` contains historical cleanup outputs — read-only reference
