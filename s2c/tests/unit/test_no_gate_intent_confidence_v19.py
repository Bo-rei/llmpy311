from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.eval.eval_system_pipeline_v19 import _predict_with_gate_disabled


class _FakeGateDetector:
    cluster_to_intent = {0: "intent_a"}


class _FakePipeline:
    gate_mode = "multisphere"
    semantic_gate_mode = "prototype"
    semantic_decision_policy = "threshold"
    gate_detector = _FakeGateDetector()
    domain_id_to_name = {0: "domain_a"}

    def _gate_predict(self, texts):
        return {
            "pred": [0 for _ in texts],
            "score": [0.0 for _ in texts],
            "distance": [0.0 for _ in texts],
            "radius": [1.0 for _ in texts],
            "margin_ok": [True for _ in texts],
            "nearest_cluster": [0 for _ in texts],
        }

    def _router_predict(self, texts, batch_size=64):
        return {
            "domain_ids": [0 for _ in texts],
            "domain_probs": [1.0 for _ in texts],
        }

    def _expert_predict_group(self, domain_name, texts, batch_size=64):
        return {
            "intent_ids": list(range(len(texts))),
            "intent_names": [f"intent_{idx}" for idx in range(len(texts))],
            "intent_probs": [0.95, 0.35][: len(texts)],
        }


def test_intent_confidence_no_gate_rejects_low_expert_confidence():
    predictions = _predict_with_gate_disabled(
        pipeline=_FakePipeline(),
        texts=["high confidence", "low confidence"],
        no_gate_mode="intent_confidence",
        router_confidence_threshold=0.5,
    )

    assert predictions[0]["final_gate_decision"] == "id"
    assert predictions[0]["intent"] == "intent_0"
    assert predictions[0]["no_gate_confidence"] == 0.95
    assert predictions[1]["final_gate_decision"] == "oos"
    assert predictions[1]["intent"] == "__oos__"
    assert predictions[1]["no_gate_confidence"] == 0.35
