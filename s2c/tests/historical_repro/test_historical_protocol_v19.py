import json
import sys
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis.historical_best_pipeline_v19 import HISTORICAL_BEST_PIPELINE


def test_profile_exposes_strict_historical_data_root():
    strict = HISTORICAL_BEST_PIPELINE.strict_replay_defaults()
    assert strict["data_root"] == "data/v19"
    assert strict["known_intents_path"] == "data/v19/KNOWN_INTENTS.json"


def test_profile_exposes_historical_gate_encoder():
    strict = HISTORICAL_BEST_PIPELINE.strict_replay_defaults()
    assert strict["gate_encoder_path"] == "all-MiniLM-L6-v2"


def test_profile_exposes_historical_target_metrics():
    strict = HISTORICAL_BEST_PIPELINE.strict_replay_defaults()
    assert strict["target_metrics"]["macro_f1"] == 0.8121241036585573
    assert strict["target_metrics"]["overall_accuracy"] == 0.8677941443898891


def test_profile_exposes_semantic_verifier_lora_shape():
    eval_defaults = HISTORICAL_BEST_PIPELINE.evaluation_defaults()
    assert eval_defaults["semantic_verifier_lora_r"] == 32
    assert eval_defaults["semantic_verifier_lora_alpha"] == 64


def test_profile_exposes_strict_historical_artifact_paths():
    strict = HISTORICAL_BEST_PIPELINE.strict_replay_defaults()
    assert strict["reference_eval_results"].endswith(
        "pipeline_v19_phase3_proto_d_narrow_t085_eval/eval_results.json"
    )
    assert strict["frozen_detector_path"].endswith(
        "gate_l2_mix2_true_lambda_1p6/detector.json"
    )


def test_historical_gate_test_split_counts_are_stable():
    data = json.loads(Path("data/v19/gate/test.json").read_text())
    id_count = sum(
        1 for row in data if row.get("intent") != "oos" and row.get("label") != 1
    )
    oos_count = len(data) - id_count
    assert len(data) == 5499
    assert id_count == 2249
    assert oos_count == 3250


def test_replay_eval_command_uses_historical_protocol():
    from tools.analysis.replay_historical_chain_v19 import build_strict_eval_command

    cmd = build_strict_eval_command(output_dir="outputs/tmp/historical_replay_eval")
    joined = " ".join(cmd)
    assert "--data_root data/v19" in joined
    assert "--gate_encoder_path all-MiniLM-L6-v2" in joined
    assert "data/multidataset/v19" not in joined


def test_bundle_classifies_frozen_baseline_wrapper_as_mainchain():
    from tools.analysis.build_historical_repro_bundle_v19 import classify_file_status

    assert classify_file_status(
        "tools/analysis/run_prototype_gate_frozen_baseline_v19.py"
    ) in {"VERIFIED_MAINCHAIN", "FROZEN_DEPENDENCY"}


def test_replay_manifest_uses_stage_order_for_historical_recovery():
    from tools.analysis.replay_historical_chain_v19 import build_replay_manifest

    manifest = build_replay_manifest(Path("outputs/reports/historical_replay_test"))
    stage_names = [stage["name"] for stage in manifest["stages"]]
    assert stage_names == [
        "truth_freeze",
        "data_validation",
        "frozen_eval_replay",
        "metric_comparison",
    ]
    frozen_stage = manifest["stages"][2]
    assert "--device" in frozen_stage["command"]
    assert "cuda" in frozen_stage["command"]


def test_bundle_render_mentions_replay_stage_status():
    from tools.analysis.build_historical_repro_bundle_v19 import render_index

    summary = {
        "strict_replay": HISTORICAL_BEST_PIPELINE.strict_replay_defaults(),
        "file_status": {},
        "replay_manifest": {
            "stages": [
                {"name": "truth_freeze", "status": "completed"},
                {"name": "metric_comparison", "status": "completed"},
            ]
        },
    }
    rendered = render_index(summary)
    assert "Replay Status" in rendered
    assert "`truth_freeze`: `completed`" in rendered


def test_multi_dataset_training_uses_gate_encoder_for_gate_stage(tmp_path):
    from tools.analysis.run_multi_dataset_training_v19 import build_stage_specs

    data_root_base = tmp_path / "data"
    data_root = data_root_base / "clinc150" / "kir25_seed42"
    data_root.mkdir(parents=True)
    (data_root / "MANIFEST.json").write_text(
        json.dumps({"domains": ["banking", "travel"]}), encoding="utf-8"
    )

    _, stage_specs = build_stage_specs(
        dataset="CLINC150",
        kir=0.25,
        seed=42,
        data_root_base=data_root_base,
        artifact_root=tmp_path / "artifacts",
        model_path="/tmp/smollm135m",
        gate_encoder_path="all-MiniLM-L6-v2",
    )

    gate_stage = next(spec for spec in stage_specs if spec.name == "gate")
    joined = " ".join(gate_stage.command)
    assert "--model_path all-MiniLM-L6-v2" in joined
    assert "/tmp/smollm135m" not in joined


def test_multi_dataset_benchmark_defaults_to_historical_gate_encoder():
    from tools.analysis.run_multi_dataset_benchmark_v19 import _resolve_gate_encoder_path

    resolved = _resolve_gate_encoder_path(
        requested_gate_encoder_path=None,
        model_path="/tmp/smollm135m",
    )
    assert resolved == "all-MiniLM-L6-v2"


def test_semantic_threshold_sweep_prefers_better_gate_working_point():
    import numpy as np
    from tools.eval.eval_system_pipeline_v19 import _tune_semantic_threshold_from_scores

    fast_gate_preds = np.asarray([1, 1, 1, 1], dtype=np.int64)
    uncertain_indices = [0, 1, 2, 3]
    semantic_scores = np.asarray([0.92, 0.89, 0.70, 0.68], dtype=np.float32)
    y_true_oos = np.asarray([0, 0, 1, 1], dtype=np.int64)

    tuning = _tune_semantic_threshold_from_scores(
        fast_gate_preds=fast_gate_preds,
        uncertain_indices=uncertain_indices,
        semantic_scores=semantic_scores,
        y_true_oos=y_true_oos,
    )

    assert tuning["best_macro_f1"] == 1.0
    assert 0.70 < tuning["best_threshold"] < 0.89
