#!/usr/bin/env python3
"""运行完整的 KIR50 × 三 seed × 四 Gate Cascade 矩阵。

该编排器只消费 ``run_cascade_components.py`` 和
``prepare_cascade_gates.py`` 生成的 manifest，不在运行时重新选择 K、阈值或
下游 checkpoint。每个单元写入自己的 ``run.log``，已有完整结果会复用，方便
GPU 任务中断后继续。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from legacy.runtime import WorkspacePaths  # noqa: E402

PATHS = WorkspacePaths.discover(PROJECT_ROOT)
ARTIFACT_ROOT = PATHS.artifact_root / "outputs" / "experiments"
MATRIX_ROOT = ARTIFACT_ROOT / "cascade_full" / "gpu_kir50"
DATA_ROOT = PATHS.prepared_data_root / "multidataset" / "v19"
DATASETS = ("clinc150", "banking77_oos", "stackoverflow")
SEEDS = (13, 42, 87)
GATES = ("frozen_k1", "frozen_selected_k", "ce_recon_selected_k", "best_controlled_baseline")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _index(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def _preflight() -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], dict[str, Any]], list[str]]:
    components = _load_json(MATRIX_ROOT / "component_plan.json")
    gates = _load_json(MATRIX_ROOT / "gates" / "gate_manifest.json")
    component_rows = _index(components["plans"], "dataset", "seed")
    gate_rows = _index(gates["units"], "dataset", "seed")
    missing: list[str] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            component = component_rows.get((dataset, seed))
            gate = gate_rows.get((dataset, seed))
            if component is None:
                missing.append(f"component plan missing: {dataset}/seed{seed}")
                continue
            if gate is None:
                missing.append(f"gate manifest missing: {dataset}/seed{seed}")
                continue
            for key in ("router", "experts"):
                if not Path(component[key]).exists():
                    missing.append(f"{dataset}/seed{seed}:{key}:{component[key]}")
            for key in ("frozen_k1", "frozen_selected_k", "ce_recon_detector", "baseline"):
                if not Path(gate[key]).is_file():
                    missing.append(f"{dataset}/seed{seed}:{key}:{gate[key]}")
    return component_rows, gate_rows, missing


def _command(
    dataset: str,
    seed: int,
    gate: str,
    component: dict[str, Any],
    gate_row: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    data_root = DATA_ROOT / dataset / f"kir50_seed{seed}"
    args = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "eval" / "eval_system_pipeline_v19.py"),
        "--model_path", str(PATHS.smollm135m),
        "--gate_encoder_path", str(PATHS.minilm),
        "--router_ckpt", str(component["router"]),
        "--experts_root", str(component["experts"]),
        "--experts_data_root", str(data_root / "experts"),
        "--data_root", str(data_root),
        "--data_root_scope", "all",
        "--output_dir", str(output_dir),
        "--device", "cuda",
        "--batch_size", "128",
        "--export_gate_diagnostics",
    ]
    if gate == "frozen_k1":
        args.extend(["--gate_mode", "multisphere", "--gate_detector_path", gate_row["frozen_k1"]])
    elif gate == "frozen_selected_k":
        args.extend(["--gate_mode", "multisphere", "--gate_detector_path", gate_row["frozen_selected_k"]])
    elif gate == "ce_recon_selected_k":
        encoder = ARTIFACT_ROOT / "minilm_representation_analysis" / "adaptation" / dataset / f"kir50_seed{seed}" / "ce_recon" / "checkpoint" / "encoder.pt"
        args.extend([
            "--gate_mode", "multisphere",
            "--gate_detector_path", gate_row["ce_recon_detector"],
            "--gate_encoder_checkpoint_path", str(encoder),
        ])
    elif gate == "best_controlled_baseline":
        args.extend(["--gate_mode", "linear_baseline", "--gate_baseline_path", gate_row["baseline"]])
    else:
        raise ValueError(f"unsupported gate: {gate}")
    return args


def _write_matrix_manifest(path: Path, payload: dict[str, Any]) -> None:
    """原子性较弱但可恢复地写入矩阵进度。

    Cascade 单元按顺序启动独立 Python 进程，单个单元可能运行数分钟。若主
    编排器在两个单元之间被中断，已有 ``eval_results.json`` 仍然有效，但旧
    实现要等全部 36 个单元结束才写 manifest，阅读者无法区分“尚未开始”和
    “已经完成一部分”。每完成一个单元就落盘一次，下一次运行仍以结果文件
    复用为准，同时把当前进度公开到 manifest。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    datasets = tuple(args.dataset or DATASETS)
    seeds = tuple(args.seed or SEEDS)
    try:
        components, gates, missing = _preflight()
    except FileNotFoundError as exc:
        raise SystemExit(f"run component/gate preparation first: {exc}") from exc
    selected_units = [
        (dataset, seed, gate)
        for dataset in datasets
        for seed in seeds
        for gate in GATES
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "cascade_full_kir50_three_seed_four_gate",
        "execute": bool(args.execute),
        "datasets": list(datasets),
        "seeds": list(seeds),
        "gates": list(GATES),
        "expected_unit_count": len(selected_units),
        "missing_components": missing,
        "units": [],
    }
    output_root = MATRIX_ROOT / "evaluations"
    output_root.mkdir(parents=True, exist_ok=True)
    if missing:
        payload["status"] = "blocked_missing_inputs"
        _write_matrix_manifest(output_root / "matrix_manifest.json", payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 2
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["CONDA_DEFAULT_ENV"] = "bo"
    payload["status"] = "running" if args.execute else "ready"
    _write_matrix_manifest(output_root / "matrix_manifest.json", payload)
    for dataset, seed, gate in selected_units:
        component = components[(dataset, seed)]
        gate_row = gates[(dataset, seed)]
        output_dir = output_root / dataset / f"kir50_seed{seed}" / gate
        result_path = output_dir / "eval_results.json"
        unit: dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "gate": gate,
            "output_dir": str(output_dir.resolve()),
        }
        if result_path.is_file():
            unit["status"] = "cached"
            payload["units"].append(unit)
            payload["completed_unit_count"] = sum(
                row["status"] in {"complete", "cached"} for row in payload["units"]
            )
            _write_matrix_manifest(output_root / "matrix_manifest.json", payload)
            continue
        if not args.execute:
            unit["status"] = "ready_to_execute"
            unit["command"] = _command(dataset, seed, gate, component, gate_row, output_dir)
            payload["units"].append(unit)
            _write_matrix_manifest(output_root / "matrix_manifest.json", payload)
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        command = _command(dataset, seed, gate, component, gate_row, output_dir)
        log_path = output_dir / "run.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(command) + "\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        unit.update({
            "status": "complete" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "log_path": str(log_path.resolve()),
        })
        payload["units"].append(unit)
        payload["completed_unit_count"] = sum(
            row["status"] in {"complete", "cached"} for row in payload["units"]
        )
        payload["status"] = "failed" if completed.returncode != 0 else "running"
        _write_matrix_manifest(output_root / "matrix_manifest.json", payload)
        if completed.returncode != 0:
            break
    completed_count = sum(row["status"] in {"complete", "cached"} for row in payload["units"])
    payload["completed_unit_count"] = completed_count
    payload["status"] = "complete" if completed_count == len(selected_units) else ("ready" if not args.execute else "failed")
    _write_matrix_manifest(output_root / "matrix_manifest.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0 if payload["status"] in {"complete", "ready"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
