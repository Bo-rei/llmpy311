#!/usr/bin/env python3
"""v20 Gate→Router→Expert 传递实验的可恢复 runner。

当前仓库可能只保留历史 eval 指标而没有 Router/Expert 权重，因此本入口先做
严格 preflight：缺少任一组件时只写 ``blocked`` 单元和待执行命令，不会把历史
结果复制成新实验，也不会为了补实验偷偷重训下游模型。
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
V19_ROOT = PROJECT_ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cluster_separability_v19"
V20_ROOT = PROJECT_ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cluster_separability_v20"
DATASETS = ("clinc150", "banking77_oos", "stackoverflow")
KIRS = (25, 50, 75)
SEED = 42
BASELINE_METHODS = ("msp", "energy", "entropy", "knn", "lof")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _selected_k(dataset: str, kir: int) -> int:
    rows = pd.read_csv(V19_ROOT / "selected_k_summary.csv")
    match = rows[
        (rows["dataset"] == dataset)
        & (rows["kir"] == kir)
        & (rows["data_seed"] == SEED)
        & (rows["distance"] == "mahalanobis_diag")
    ]
    if len(match) != 1:
        raise ValueError(f"selected K not unique for {dataset}/KIR{kir}")
    return int(match.iloc[0]["selected_k"])


def _best_validation_baseline(dataset: str, kir: int) -> str:
    rows = pd.read_csv(V19_ROOT / "gate_baseline_by_seed.csv")
    rows = rows[
        (rows["dataset"] == dataset)
        & (rows["kir"] == kir)
        & (rows["data_seed"] == SEED)
        & rows["method"].isin(BASELINE_METHODS)
    ]
    if rows.empty:
        raise ValueError(f"no controlled baseline for {dataset}/KIR{kir}")
    return str(rows.sort_values("validation_oos_f1", ascending=False).iloc[0]["method"])


def _paths(overrides: dict[str, str] | None = None) -> dict[str, Path]:
    """集中定义可由环境变量/CLI 替换的下游组件路径。"""

    paths = {
        "eval_script": PROJECT_ROOT / "tools" / "eval" / "eval_system_pipeline_v19.py",
        "model_path": PROJECT_ROOT.parent / "smollm135m",
        "router_ckpt": PROJECT_ROOT / "outputs" / "experiments" / "components" / "router" / "router_v19" / "best_model.pt",
        "experts_root": PROJECT_ROOT / "outputs" / "experiments" / "components" / "experts" / "experts_v19",
        "data_root": PROJECT_ROOT.parent / "assets" / "datasets" / "s2c" / "prepared" / "data" / "multidataset" / "v19",
    }
    for key, value in (overrides or {}).items():
        if value:
            paths[key] = Path(value)
    return paths


def build_matrix(v20_root: Path = V20_ROOT, path_overrides: dict[str, str] | None = None) -> dict[str, Any]:
    paths = _paths(path_overrides)
    rows: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for kir in KIRS:
            selected = _selected_k(dataset, kir)
            baseline = _best_validation_baseline(dataset, kir)
            gates = [
                ("k1", V19_ROOT / "fixed" / dataset / f"kir{kir}_seed{SEED}" / "mahalanobis_diag" / "k1" / "detector.json"),
                ("k2", V19_ROOT / "fixed" / dataset / f"kir{kir}_seed{SEED}" / "mahalanobis_diag" / "k2" / "detector.json"),
                ("selected_k", V19_ROOT / "tuned" / dataset / f"kir{kir}_seed{SEED}" / "mahalanobis_diag" / f"k{selected}" / "detector.json"),
                (f"baseline_{baseline}", None),
            ]
            for gate_name, detector in gates:
                out = v20_root / "end_to_end" / dataset / f"kir{kir}_seed{SEED}" / gate_name
                missing = [
                    str(path)
                    for path in paths.values()
                    if not path.exists()
                ]
                if detector is not None and not detector.is_file():
                    missing.append(str(detector))
                status = "ready" if not missing else "blocked_missing_inputs"
                command = [
                    "python", str(paths["eval_script"]),
                    "--gate_detector_path", str(detector) if detector else "<baseline-gate-adapter>",
                    "--data_root", str(paths["data_root"] / dataset / f"kir{kir}_seed{SEED}"),
                    "--data_root_scope", "all",
                    "--output_dir", str(out),
                    "--gate_mode", "multisphere",
                    "--semantic_gate_mode", "none",
                    "--batch_size", "64",
                ]
                rows.append({
                    "dataset": dataset,
                    "kir": kir,
                    "data_seed": SEED,
                    "gate": gate_name,
                    "selected_k": selected,
                    "selected_validation_baseline": baseline,
                    "status": status,
                    "missing_count": len(missing),
                    "missing_inputs": "|".join(missing),
                    "output_dir": str(out),
                })
                commands.append({"dataset": dataset, "kir": kir, "gate": gate_name, "command": shlex.join(command)})
    _write_csv(v20_root / "end_to_end_matrix.csv", rows)
    _write_json(v20_root / "end_to_end_commands.json", {"commands": commands, "paths": {k: str(v) for k, v in paths.items()}})
    payload = {
        "protocol": "cluster_separability_v20_end_to_end",
        "expected_units": len(rows),
        "ready_units": sum(row["status"] == "ready" for row in rows),
        "blocked_units": sum(row["status"] != "ready" for row in rows),
        "reason": "component preflight only; no historical eval result is copied",
        "v19_frozen": True,
    }
    _write_json(v20_root / "end_to_end_preflight.json", payload)
    return payload


def execute_ready(v20_root: Path = V20_ROOT) -> dict[str, int]:
    """执行已通过 preflight 的命令；默认不调用，避免隐式重训或跨设备运行。"""

    matrix = pd.read_csv(v20_root / "end_to_end_matrix.csv")
    completed = 0
    failed = 0
    for _, row in matrix[matrix["status"] == "ready"].iterrows():
        command_row = next(
            item for item in json.loads((v20_root / "end_to_end_commands.json").read_text())["commands"]
            if item["dataset"] == row["dataset"] and int(item["kir"]) == int(row["kir"]) and item["gate"] == row["gate"]
        )
        result = subprocess.run(command_row["command"], shell=True, cwd=PROJECT_ROOT, check=False)
        if result.returncode == 0:
            completed += 1
        else:
            failed += 1
    return {"completed": completed, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare/execute v20 end-to-end Gate transfer")
    parser.add_argument("--v20-root", default=str(V20_ROOT))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--router-ckpt", default=None)
    parser.add_argument("--experts-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--eval-script", default=None)
    parser.add_argument("--execute-ready", action="store_true")
    args = parser.parse_args()
    root = Path(args.v20_root)
    payload = build_matrix(root, {
        "model_path": args.model_path,
        "router_ckpt": args.router_ckpt,
        "experts_root": args.experts_root,
        "data_root": args.data_root,
        "eval_script": args.eval_script,
    })
    if args.execute_ready and payload["ready_units"]:
        payload["execution"] = execute_ready(root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
