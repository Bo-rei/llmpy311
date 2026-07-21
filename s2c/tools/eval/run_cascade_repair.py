#!/usr/bin/env python3
"""在固定 Router/Expert 上重跑代表性 Gate→Router→Expert。

该入口只负责实验编排，不训练模型。它明确把「修复后的下游模型」和四个
代表性 Gate 组合起来，并为每个单元留下矩阵 manifest；历史 smoke 结果不会
被覆盖。默认配置对应当前论文收口所需的 3 个数据集、KIR50、seed42。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import WorkspacePaths  # noqa: E402

PATHS = WorkspacePaths.discover(PROJECT_ROOT)
DATA_ROOT = PATHS.prepared_data_root / "multidataset" / "v19"
ARTIFACT_ROOT = PATHS.artifact_root / "outputs" / "experiments"
COMPONENT_ROOT = ARTIFACT_ROOT / "components"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "cascade_repair" / "gpu_kir50_seed42" / "evaluations"
DEFAULT_REPAIRED_EXPERTS = ARTIFACT_ROOT / "cascade_repair" / "gpu_kir50_seed42" / "expert_models"
DATASETS = ("clinc150", "banking77_oos", "stackoverflow")


def selected_k(dataset: str, kir: int, seed: int) -> int:
    """读取已冻结的 validation-selected K；不在 test 上重新选择。"""

    summary = pd.read_csv(ARTIFACT_ROOT / "cluster_separability_v19" / "selected_k_summary.csv")
    rows = summary[
        summary["dataset"].eq(dataset)
        & summary["kir"].eq(kir)
        & summary["data_seed"].eq(seed)
        & summary["distance"].eq("mahalanobis_diag")
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one selected K for {dataset}/kir{kir}/seed{seed}")
    return int(rows.iloc[0]["selected_k"])


def component_paths(
    dataset: str,
    kir: int,
    seed: int,
    repaired_experts_root: Path,
) -> dict[str, Path]:
    data_root = DATA_ROOT / dataset / f"kir{kir}_seed{seed}"
    # CLINC 的既有 Expert 已通过单点检查；Banking/StackOverflow 使用本轮 GPU
    # 重训产物。这样不会把一套低质量 smoke checkpoint 错用于全部数据集。
    if dataset == "clinc150":
        experts = COMPONENT_ROOT / "experts" / f"{dataset}_kir{kir}_seed{seed}_experts_v19"
    else:
        # evaluator 需要传入包含 ``banking/`` 或 ``stackoverflow/`` 子目录
        # 的 experts_root，而不是直接传入某个 domain 目录。
        experts = repaired_experts_root
    return {
        "data_root": data_root,
        "router": COMPONENT_ROOT / "router" / f"{dataset}_kir{kir}_seed{seed}_router_v19" / "best_model.pt",
        "experts": experts,
    }


def gate_specs(dataset: str, kir: int, seed: int) -> dict[str, dict[str, Any]]:
    k = selected_k(dataset, kir, seed)
    fixed_root = ARTIFACT_ROOT / "cluster_separability_v19" / "fixed" / dataset / f"kir{kir}_seed{seed}" / "mahalanobis_diag"
    gate_root = COMPONENT_ROOT / "gates"
    adaptation = (
        ARTIFACT_ROOT
        / "minilm_representation_analysis"
        / "adaptation"
        / dataset
        / f"kir{kir}_seed{seed}"
        / "ce_recon"
        / "checkpoint"
        / "encoder.pt"
    )
    return {
        "frozen_k1": {"gate_mode": "multisphere", "detector": fixed_root / "k1" / "detector.json"},
        "frozen_selected_k": {
            "gate_mode": "multisphere",
            "detector": fixed_root / f"k{k}" / "detector.json",
        },
        "ce_recon_selected_k": {
            "gate_mode": "multisphere",
            "detector": gate_root / f"{dataset}_kir{kir}_seed{seed}_ce_recon_frozen_selected_k.detector.json",
            "encoder_checkpoint": adaptation,
        },
        "best_controlled_baseline": {
            "gate_mode": "linear_baseline",
            "baseline": gate_root / f"{dataset}_kir{kir}_seed{seed}_best_linear_baseline.pkl",
        },
    }


def preflight(dataset: str, kir: int, seed: int, repaired_experts_root: Path) -> dict[str, Any]:
    components = component_paths(dataset, kir, seed, repaired_experts_root)
    missing = [str(path) for path in components.values() if not path.exists()]
    # ``experts_root`` 可能同时保存 Banking 和 StackOverflow 两个修复产物；
    # 仅检查 ``glob("*/best_model.pt")`` 会把“有其它数据集 Expert”误判为
    # 当前数据集可用。这里按数据协议明确检查所需 domain，避免评估启动后
    # 才在 ``_ensure_active_expert`` 阶段失败。
    # 每个数据准备目录的 MANIFEST 声明所需 domain；CLINC 有多个 Expert，
    # Banking/StackOverflow 各只有一个。不要把“整个 experts_root 至少有
    # 一个 checkpoint”当作充分条件。
    expected_domains: list[str] = []
    manifest_path = components["data_root"] / "MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_domains = [str(value) for value in manifest.get("domains", [])]
    if not expected_domains:
        expected_domains = {
            "banking77_oos": ["banking"],
            "stackoverflow": ["stackoverflow"],
        }.get(dataset, [])
    if not expected_domains:
        missing.append(f"{dataset}:cannot resolve expected Expert domains")
    specs = gate_specs(dataset, kir, seed)
    for name, spec in specs.items():
        for key in ("detector", "encoder_checkpoint", "baseline"):
            value = spec.get(key)
            if value is not None and not Path(value).is_file():
                missing.append(f"{name}:{value}")
        if spec["gate_mode"] == "multisphere":
            for domain in expected_domains:
                expert_checkpoint = components["experts"] / domain / "best_model.pt"
                if not expert_checkpoint.is_file():
                    missing.append(
                        f"{name}:missing expert checkpoint for domain={domain} "
                        f"under {components['experts']}"
                    )
    return {
        "status": "ready" if not missing else "blocked_missing_components",
        "dataset": dataset,
        "kir": kir,
        "data_seed": seed,
        "components": {key: str(value.resolve()) for key, value in components.items()},
        "gates": {
            name: {
                key: str(value.resolve()) if isinstance(value, Path) else value
                for key, value in spec.items()
            }
            for name, spec in specs.items()
        },
        "missing": missing,
    }


def run_dataset(
    dataset: str,
    kir: int,
    seed: int,
    output_root: Path,
    repaired_experts_root: Path,
    execute: bool,
) -> dict[str, Any]:
    audit = preflight(dataset, kir, seed, repaired_experts_root)
    if not execute:
        audit["status"] = "preflight_only" if not audit["missing"] else audit["status"]
        return audit
    if audit["missing"]:
        return audit

    components = component_paths(dataset, kir, seed, repaired_experts_root)
    data_root = components["data_root"]
    rows: list[dict[str, Any]] = []
    for gate_name, spec in gate_specs(dataset, kir, seed).items():
        output_dir = output_root / dataset / f"kir{kir}_seed{seed}" / gate_name
        result_path = output_dir / "eval_results.json"
        if result_path.is_file():
            rows.append({"gate": gate_name, "status": "cached", "output_dir": str(output_dir.resolve())})
            continue

        command = [
            sys.executable,
            str(PROJECT_ROOT / "tools" / "eval" / "eval_system_pipeline_v19.py"),
            "--model_path", str(PATHS.smollm135m),
            "--gate_encoder_path", str(PATHS.minilm),
            "--gate_mode", str(spec["gate_mode"]),
            "--router_ckpt", str(components["router"]),
            "--experts_root", str(components["experts"]),
            "--experts_data_root", str(data_root / "experts"),
            "--data_root", str(data_root),
            "--data_root_scope", "all",
            "--output_dir", str(output_dir),
            "--device", "cuda",
            "--batch_size", "128",
            "--export_gate_diagnostics",
        ]
        if spec["gate_mode"] == "multisphere":
            command.extend(["--gate_detector_path", str(spec["detector"])])
        elif spec["gate_mode"] == "linear_baseline":
            command.extend(["--gate_baseline_path", str(spec["baseline"])])
        if spec.get("encoder_checkpoint"):
            command.extend(["--gate_encoder_checkpoint_path", str(spec["encoder_checkpoint"])])

        env = dict(os.environ)
        env["CONDA_DEFAULT_ENV"] = "bo"
        # 不把长时间推理的 stdout 放进 PIPE：日志超过系统 pipe buffer 时，
        # 子进程会阻塞而父进程也无法收到返回码。每个单元写自己的 run.log，
        # 既可审计又支持中断后从已有 eval_results.json 继续。
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "run.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        rows.append(
            {
                "gate": gate_name,
                "status": "complete" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "output_dir": str(output_dir.resolve()),
                "log_path": str(log_path.resolve()),
            }
        )
        if completed.returncode != 0:
            break

    return {
        "status": "complete" if all(row["status"] in {"complete", "cached"} for row in rows) else "failed",
        "dataset": dataset,
        "units": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--kir", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repaired-experts-root", type=Path, default=DEFAULT_REPAIRED_EXPERTS)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    datasets = args.dataset or list(DATASETS)
    args.output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "protocol": "representative_cascade_repair_fixed_downstream",
        "execute": bool(args.execute),
        "project_root": str(PROJECT_ROOT.resolve()),
        "python": sys.executable,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "gpu_requested": bool(args.execute),
        "datasets": [
            run_dataset(
                dataset,
                args.kir,
                args.seed,
                args.output_root,
                args.repaired_experts_root,
                args.execute,
            )
            for dataset in datasets
        ],
    }
    (args.output_root / "matrix_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if all(item["status"] in {"complete", "preflight_only"} for item in payload["datasets"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
