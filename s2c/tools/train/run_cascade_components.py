#!/usr/bin/env python3
"""为完整 KIR50 Cascade 矩阵准备 Router/Expert 组件。

本脚本只负责组件训练和 provenance，不负责 Gate 或最终评价。它把已有的
seed42 组件作为稳定参考复用，对 seed13/87 只训练缺失组件。这样可以避免把
同一个 SmolLM checkpoint 隐式复制到多个结果目录，也能让后续 evaluator
直接读取一份明确的 ``component_manifest.json``。

固定协议：
    dataset ∈ {clinc150, banking77_oos, stackoverflow}
    kir = 50
    seed ∈ {13, 42, 87}

运行前请在 ``bo`` 环境中设置 CUDA 动态库路径。默认只做 preflight；传入
``--execute`` 才会启动训练。已有 ``best_model.pt`` 会被复用，不覆盖历史
产物。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime import WorkspacePaths  # noqa: E402

PATHS = WorkspacePaths.discover(PROJECT_ROOT)
DATA_ROOT = PATHS.prepared_data_root / "multidataset" / "v19"
ARTIFACT_ROOT = PATHS.artifact_root / "outputs" / "experiments"
OUTPUT_ROOT = ARTIFACT_ROOT / "cascade_full" / "gpu_kir50"
DATASETS = ("clinc150", "banking77_oos", "stackoverflow")
SEEDS = (13, 42, 87)

ROUTER_SCRIPT = PROJECT_ROOT / "tools" / "train" / "train_router_v19.py"
EXPERT_BATCH_SCRIPT = PROJECT_ROOT / "tools" / "train" / "train_all_experts_v19.py"
EXPERT_SINGLE_SCRIPT = PROJECT_ROOT / "tools" / "train" / "train_expert_v19.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_root(dataset: str, seed: int) -> Path:
    return DATA_ROOT / dataset / f"kir50_seed{seed}"


def _seed_output(seed: int) -> Path:
    return OUTPUT_ROOT / f"seed{seed}" / "downstream"


def _existing_seed42_paths(dataset: str) -> dict[str, Any]:
    """返回已审计 seed42 组件；不复制大 checkpoint。"""

    router = ARTIFACT_ROOT / "components" / "router" / f"{dataset}_kir50_seed42_router_v19" / "best_model.pt"
    if dataset == "clinc150":
        experts = ARTIFACT_ROOT / "components" / "experts" / f"{dataset}_kir50_seed42_experts_v19"
    else:
        experts = ARTIFACT_ROOT / "cascade_repair" / "gpu_kir50_seed42" / "expert_models"
    return {"router": router, "experts": experts, "source": "existing_seed42_audited"}


def _domains(dataset_root: Path) -> list[str]:
    """返回该数据协议真正声明的 Expert domain。

    不能直接枚举 ``experts/`` 下的所有目录：StackOverflow 的准备目录中
    还保留了其它实验流程的 ``data_backend``、``mobile`` 等目录，但本轮
    ``MANIFEST.json`` 明确声明它只有一个 ``stackoverflow`` domain。旧实现
    用 ``sorted(...)[0]`` 会因此训练错误的 ``data_backend`` 二分类 Expert，
    后续 evaluator 又按 ``stackoverflow`` 查找 checkpoint，最终表现为
    ``missing_components`` 或隐蔽的标签空间错位。

    MANIFEST 是数据协议的唯一事实来源；仅在旧数据没有 MANIFEST 时，才
    回退到目录枚举，保持历史数据可读性。返回前还会验证每个声明的目录
    存在且包含 train/val/test 三个 split，尽早暴露数据准备错误。
    """

    experts_root = dataset_root / "experts"
    manifest_path = dataset_root / "MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared = [str(value) for value in manifest.get("domains", [])]
        if not declared:
            # 某些早期 manifest 没有顶层 domains；从 domain_map 保持协议兼容。
            declared = [str(value) for value in manifest.get("domain_map", {}).keys()]
        if not declared:
            raise ValueError(f"MANIFEST has no declared domains: {manifest_path}")
        missing = []
        for domain in declared:
            domain_root = experts_root / domain
            if not domain_root.is_dir():
                missing.append(str(domain_root))
                continue
            missing.extend(
                str(domain_root / f"{split}.json")
                for split in ("train", "val", "test")
                if not (domain_root / f"{split}.json").is_file()
            )
        if missing:
            raise FileNotFoundError(
                "Manifest-declared Expert domains are incomplete: "
                + ", ".join(missing)
            )
        return sorted(dict.fromkeys(declared))

    if not experts_root.is_dir():
        raise FileNotFoundError(f"Expert data root does not exist: {experts_root}")
    return sorted(path.name for path in experts_root.iterdir() if path.is_dir())


def _planned_component(dataset: str, seed: int) -> dict[str, Any]:
    data_root = _dataset_root(dataset, seed)
    if seed == 42:
        paths = _existing_seed42_paths(dataset)
        return {
            "dataset": dataset,
            "seed": seed,
            "data_root": data_root,
            "router": paths["router"],
            "experts": paths["experts"],
            "source": paths["source"],
            "domains": _domains(data_root),
            "commands": [],
        }

    root = _seed_output(seed) / dataset
    domains = _domains(data_root)
    # BANKING77-OOS 与 StackOverflow 都只有一个 domain。单域 Router
    # 没有可学习的路由决策；继续实例化 SmolLM 并训练一个 1-class head
    # 只会浪费显存，甚至在大 batch 下触发 OOM。用小型协议文件标记
    # constant router，由推理管线直接返回唯一 domain。
    single_domain = len(domains) == 1
    router = root / "router" / ("constant_router.json" if single_domain else "best_model.pt")
    experts = root / "experts"
    commands: list[dict[str, Any]] = []
    if not single_domain:
        router_command = [
            sys.executable,
            str(ROUTER_SCRIPT),
            "--model_path", str(PATHS.smollm135m),
            "--data_dir", str(data_root / "router"),
            "--output_dir", str(router.parent),
            "--epochs", "10",
            "--batch_size", "32",
            "--patience", "5",
            "--num_workers", "0",
            "--seed", str(seed),
        ]
        # 已有 checkpoint 只做 provenance 复用，不重复训练同一组件。
        if not router.is_file():
            commands.append({"kind": "router", "command": router_command})
    if dataset == "clinc150":
        expert_command = [
            sys.executable,
            str(EXPERT_BATCH_SCRIPT),
            "--domains", *domains,
            "--model_path", str(PATHS.smollm135m),
            "--data_dir", str(data_root / "experts"),
            "--output_dir", str(experts),
            "--epochs", "15",
            "--batch_size", "32",
            "--patience", "5",
            "--num_workers", "0",
            "--seed", str(seed),
            "--skip_existing",
        ]
        if any(not (experts / domain / "best_model.pt").is_file() for domain in domains):
            commands.append({"kind": "experts", "command": expert_command})
    else:
        domain = domains[0]
        expert_command = [
            sys.executable,
            str(EXPERT_SINGLE_SCRIPT),
            "--domain", domain,
            "--model_path", str(PATHS.smollm135m),
            "--data_dir", str(data_root / "experts"),
            "--output_dir", str(experts),
            "--epochs", "15",
            "--batch_size", "32",
            "--patience", "5",
            "--num_workers", "0",
            "--seed", str(seed),
        ]
        if not (experts / domain / "best_model.pt").is_file():
            commands.append({"kind": "experts", "command": expert_command})
    return {
        "dataset": dataset,
        "seed": seed,
        "data_root": data_root,
        "router": router,
        "experts": experts,
        "source": "gpu_trained_for_cascade_matrix",
        "router_mode": "constant" if single_domain else "learned",
        "domains": domains,
        "commands": commands,
    }


def _audit_plan(plan: dict[str, Any], previous: dict[str, Any] | None = None) -> dict[str, Any]:
    """检查组件完整性，并尽量复用未变化 checkpoint 的哈希。

    全矩阵包含多个百 MB 级权重。每次只想做 preflight 时重新读取全部权重会
    把几分钟浪费在磁盘 I/O 上；因此 manifest 同时保存文件大小和修改时间，
    只有文件元数据改变或旧 manifest 没有这些字段时才重新计算 SHA256。
    这不会放宽完整性检查：路径仍每次检查，元数据变化仍会触发新哈希。
    """
    missing: list[str] = []
    data_root = Path(plan["data_root"])
    for path in (data_root / "router" / "train.json", data_root / "gate" / "test.json"):
        if not path.is_file():
            missing.append(str(path))
    router = Path(plan["router"])
    if not router.is_file():
        missing.append(str(router))
    experts = Path(plan["experts"])
    for domain in plan["domains"]:
        checkpoint = experts / domain / "best_model.pt"
        if not checkpoint.is_file():
            missing.append(str(checkpoint))
    checkpoint_hashes: dict[str, str] = {}
    checkpoint_stat: dict[str, dict[str, int]] = {}
    previous_hashes = (previous or {}).get("checkpoint_sha256", {})
    previous_stat = (previous or {}).get("checkpoint_stat", {})

    def record_hash(key: str, path: Path) -> None:
        if not path.is_file():
            return
        stat = path.stat()
        current_stat = {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
        checkpoint_stat[key] = current_stat
        if previous_hashes.get(key) and previous_stat.get(key) == current_stat:
            checkpoint_hashes[key] = str(previous_hashes[key])
        else:
            checkpoint_hashes[key] = _sha256(path)

    if router.is_file():
        record_hash("router", router)
    for domain in plan["domains"]:
        checkpoint = experts / domain / "best_model.pt"
        if checkpoint.is_file():
            record_hash(f"expert:{domain}", checkpoint)
    return {
        "dataset": plan["dataset"],
        "seed": plan["seed"],
        "status": "ready" if not missing else "missing_components",
        "missing": missing,
        "router": str(router.resolve()),
        "experts": str(experts.resolve()),
        "domains": plan["domains"],
        "source": plan["source"],
        "checkpoint_sha256": checkpoint_hashes,
        "checkpoint_stat": checkpoint_stat,
    }


def _run_plan(plan: dict[str, Any]) -> None:
    if plan["source"] == "existing_seed42_audited":
        return
    output_dir = _seed_output(int(plan["seed"])) / str(plan["dataset"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "component_training.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["CONDA_DEFAULT_ENV"] = "bo"
    with log_path.open("a", encoding="utf-8") as log:
        if plan.get("router_mode") == "constant":
            router_path = Path(plan["router"])
            router_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 1,
                "router_mode": "constant",
                "dataset": str(plan["dataset"]),
                "seed": int(plan["seed"]),
                "domain": str(plan["domains"][0]),
                "domain_label": 0,
                "reason": "single-domain dataset; no learned routing decision is required",
            }
            router_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            log.write("\nconstant router: " + str(router_path) + "\n")
            log.flush()
        for item in plan["commands"]:
            command = [str(value) for value in item["command"]]
            log.write("\n$ " + " ".join(command) + "\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"{item['kind']} training failed with code {completed.returncode}")


def _plan_payload(plan: dict[str, Any]) -> dict[str, Any]:
    """把内部 Path 对象转成 manifest 可读的稳定 JSON。"""

    return {
        **{
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in plan.items()
            if key != "commands"
        },
        "commands": [
            {"kind": item["kind"], "command": [str(value) for value in item["command"]]}
            for item in plan["commands"]
        ],
    }


def _merge_plan_records(
    current: list[dict[str, Any]],
    existing_path: Path,
) -> list[dict[str, Any]]:
    """按 dataset/seed 合并部分调用，避免覆盖其它已准备组件。

    编排器常被分批调用（例如先补 StackOverflow，再补 CLINC）。如果每次
    只写本次参数，矩阵 preflight 会看不到先前的组件。合并只作用于小型
    manifest，不复制或修改任何 checkpoint。
    """

    merged: dict[tuple[str, int], dict[str, Any]] = {}
    if existing_path.is_file():
        try:
            previous = json.loads(existing_path.read_text(encoding="utf-8"))
            for plan in previous.get("plans", []):
                if "dataset" in plan and "seed" in plan:
                    merged[(str(plan["dataset"]), int(plan["seed"]))] = plan
        except (OSError, ValueError, TypeError):
            # 损坏的旧 manifest 不应阻塞重新生成；当前请求会写出完整记录。
            merged = {}
    for plan in current:
        merged[(str(plan["dataset"]), int(plan["seed"]))] = _plan_payload(plan)
    return [merged[key] for key in sorted(merged)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    datasets = tuple(args.dataset or DATASETS)
    seeds = tuple(args.seed or SEEDS)
    plans = [_planned_component(dataset, seed) for dataset in datasets for seed in seeds]
    plan_path = OUTPUT_ROOT / "component_plan.json"
    previous_audits: dict[tuple[str, int], dict[str, Any]] = {}
    if plan_path.is_file():
        try:
            previous_payload = json.loads(plan_path.read_text(encoding="utf-8"))
            for audit in previous_payload.get("audits_after_run", []) + previous_payload.get("audits_before_run", []):
                if "dataset" in audit and "seed" in audit:
                    previous_audits[(str(audit["dataset"]), int(audit["seed"]))] = audit
        except (OSError, ValueError, TypeError):
            previous_audits = {}
    audits = [
        _audit_plan(plan, previous_audits.get((str(plan["dataset"]), int(plan["seed"]))))
        for plan in plans
    ]
    merged_plan_records = _merge_plan_records(plans, plan_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "cascade_full_kir50_downstream_components",
        "execute": bool(args.execute),
        "requested_datasets": list(datasets),
        "requested_seeds": list(seeds),
        "datasets": sorted({str(plan["dataset"]) for plan in merged_plan_records}),
        "seeds": sorted({int(plan["seed"]) for plan in merged_plan_records}),
        "plans": merged_plan_records,
        "audits_before_run": audits,
        "manifest_merge": "preserve existing dataset/seed plans and replace requested keys",
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    if args.execute:
        for plan, audit in zip(plans, audits):
            if audit["missing"] and plan["source"] == "existing_seed42_audited":
                raise RuntimeError(f"seed42 audited components missing: {audit['missing']}")
            _run_plan(plan)
        audits = [_audit_plan(plan, previous_audits.get((str(plan["dataset"]), int(plan["seed"])))) for plan in plans]
        payload["audits_after_run"] = audits
        payload["status"] = "complete" if all(item["status"] == "ready" for item in audits) else "incomplete"
        # 重新合并一次：当前调用可能刚刚生成了 constant router 或 Expert，
        # 其它 dataset/seed 的计划仍需保留在同一份矩阵 manifest 中。
        payload["plans"] = _merge_plan_records(plans, plan_path)
        payload["datasets"] = sorted({str(plan["dataset"]) for plan in payload["plans"]})
        payload["seeds"] = sorted({int(plan["seed"]) for plan in payload["plans"]})
        plan_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
        )
    else:
        payload["status"] = "preflight_only"
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0 if not args.execute or payload.get("status") == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
