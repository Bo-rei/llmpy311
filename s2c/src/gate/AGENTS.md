# src/gate/ — OOS Detection Gate Modules

## OVERVIEW
Multiple OOS (Out-of-Scope) detection strategies: multi-sphere SVDD, multi-prototype cosine similarity, LLM-based semantic verification, and dual-representation fusion.

## STRUCTURE
```
gate/
├── multi_sphere_oos_detector.py    # K-means hyperspheres, quantile radius, Mahalanobis support
├── multi_prototype_gate.py         # Per-intent K-means prototypes, two-threshold scoring
├── llm_semantic_verifier.py        # Next-token Yes/No likelihood for semantic ID check
├── llm_uncertainty_verifier.py     # Versioned prompt template (v1) for uncertainty estimation
├── intent_prototype_matcher.py     # SmolLM-space prototype building + similarity scoring
├── dual_representation_gate.py     # MiniLM + SmolLM weighted fusion (alpha + beta = 1.0)
└── contrastive_intent_encoder.py   # Projection head + SupCon / hard-negative contrastive loss
```

## WHERE TO LOOK
| Task | File | Notes |
|------|------|-------|
| Multi-sphere OOS detection | `multi_sphere_oos_detector.py` | `MultiSphereOOSDetector.fit()` → `predict_with_scores()` |
| Multi-prototype scoring | `multi_prototype_gate.py` | `MultiPrototypeGate.from_dict()` for deserialization |
| LLM semantic verification | `llm_semantic_verifier.py` | Uses router model's `.base` for logits |
| Dual-rep fusion | `dual_representation_gate.py` | Enforces alpha + beta == 1.0 |
| Contrastive training | `contrastive_intent_encoder.py` | `supervised_contrastive_loss()`, `hard_negative_contrastive_loss()` |

## CONVENTIONS
- All gate classes use `@dataclass` for config objects
- Serialization via `to_dict()` / `from_dict()` pattern
- Score modes: `max`, `top2_margin`, `top2_margin_conf`
- Two-threshold decision: `tau_low` (reject), `tau_high` (accept), between = uncertain

## ANTI-PATTERNS
- **DO NOT** modify `multi_sphere_oos_detector.py` radius computation without validating on Val set
- **DO NOT** change prompt templates in `llm_semantic_verifier.py` / `llm_uncertainty_verifier.py` without versioning
- **DO NOT** use `torch.no_grad()` in contrastive training loops — breaks gradient flow

## NOTES
- `multi_sphere_oos_detector.py` supports 3 center modes: `kmeans`, `class_centroid`, `class_centroid_mixture`
- Distance metrics: `euclidean` or `mahalanobis_diag`
- `dual_representation_gate.py` is standalone — not currently wired into the main pipeline
