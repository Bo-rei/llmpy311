"""完整 Cascade 矩阵编排器的轻量协议测试。

这些测试不加载 SmolLM，也不启动 GPU；它们只锁定矩阵维度和命令行中最容易
被误配的 Gate/encoder 参数。
"""

from pathlib import Path
import json

from tools.eval.run_cascade_matrix import _command
from tools.train.run_cascade_components import _domains, _planned_component


def _component() -> dict[str, str]:
    return {"router": "/tmp/router.pt", "experts": "/tmp/experts"}


def _gates() -> dict[str, str]:
    return {
        "frozen_k1": "/tmp/k1.json",
        "frozen_selected_k": "/tmp/selected.json",
        "ce_recon_detector": "/tmp/ce.json",
        "baseline": "/tmp/baseline.pkl",
    }


def test_matrix_command_uses_frozen_detector() -> None:
    command = _command("clinc150", 13, "frozen_k1", _component(), _gates(), Path("/tmp/out"))
    assert "--gate_mode" in command
    assert command[command.index("--gate_mode") + 1] == "multisphere"
    assert command[command.index("--gate_detector_path") + 1] == "/tmp/k1.json"
    assert "--gate_encoder_checkpoint_path" not in command


def test_matrix_command_adds_adapted_encoder_only_for_ce_recon() -> None:
    command = _command("banking77_oos", 87, "ce_recon_selected_k", _component(), _gates(), Path("/tmp/out"))
    assert command[command.index("--gate_detector_path") + 1] == "/tmp/ce.json"
    checkpoint = command[command.index("--gate_encoder_checkpoint_path") + 1]
    assert checkpoint.endswith("banking77_oos/kir50_seed87/ce_recon/checkpoint/encoder.pt")


def test_matrix_command_uses_linear_baseline_without_detector() -> None:
    command = _command("stackoverflow", 42, "best_controlled_baseline", _component(), _gates(), Path("/tmp/out"))
    assert command[command.index("--gate_mode") + 1] == "linear_baseline"
    assert command[command.index("--gate_baseline_path") + 1] == "/tmp/baseline.pkl"
    assert "--gate_detector_path" not in command


def test_domains_follow_manifest_instead_of_residual_expert_directories(tmp_path: Path) -> None:
    """StackOverflow 的残留目录不能被误当成真实 domain。"""

    root = tmp_path / "dataset"
    (root / "experts" / "stackoverflow").mkdir(parents=True)
    (root / "experts" / "data_backend").mkdir(parents=True)
    for split in ("train", "val", "test"):
        (root / "experts" / "stackoverflow" / f"{split}.json").write_text("[]")
    (root / "MANIFEST.json").write_text(json.dumps({"domains": ["stackoverflow"]}))

    assert _domains(root) == ["stackoverflow"]


def test_single_domain_component_uses_constant_router() -> None:
    plan = _planned_component("stackoverflow", 13)
    assert plan["router_mode"] == "constant"
    assert plan["domains"] == ["stackoverflow"]
    assert all(item["kind"] != "router" for item in plan["commands"])
