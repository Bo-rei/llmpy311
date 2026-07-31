#!/usr/bin/env python3
"""Launch the pinned official MOGB entrypoint through an auditable shim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


SCRIPT_PATH = Path(__file__).resolve()
S2C_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG_PATH = S2C_ROOT / "configs" / "baselines" / "mogb_official_modernized_smoke.yaml"
DEFAULT_SOURCE_ROOT = S2C_ROOT / "data" / "sources" / "textoir" / "dffe2b1b848a069a6808f8089b4cb9bd16e2062b"
MANIFEST_NAME = "manifest.json"
TRANSIENT_HASH_PARTS = {".git", "__pycache__", ".pytest_cache"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML in {path}")
    return data


def resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (S2C_ROOT / path).resolve()


def file_or_tree_hash(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "sha256": None, "file_count": 0}
    if path.is_file():
        return {
            "path": str(path),
            "exists": True,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "file_count": 1,
        }
    digest = hashlib.sha256()
    file_count = 0
    for file_path in sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and not TRANSIENT_HASH_PARTS.intersection(candidate.relative_to(path).parts)
        and candidate.suffix != ".pyc"
    ):
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
        file_count += 1
    return {
        "path": str(path),
        "exists": True,
        "sha256": digest.hexdigest(),
        "file_count": file_count,
    }


def copy_dataset_snapshot(source_dir: Path, target_dir: Path) -> dict[str, Any]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source dataset directory does not exist: {source_dir}")
    source_hash = file_or_tree_hash(source_dir)
    if target_dir.exists() and file_or_tree_hash(target_dir).get("sha256") == source_hash.get("sha256"):
        target_hash = file_or_tree_hash(target_dir)
        return {
            "source_dir": str(source_dir),
            "target_dir": str(target_dir),
            "copied": False,
            "hash_verified": source_hash.get("sha256") == target_hash.get("sha256"),
            "source_hash": source_hash,
            "target_hash": target_hash,
        }

    temp_dir = target_dir.parent / f".{target_dir.name}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    for file_path in sorted(candidate for candidate in source_dir.rglob("*") if candidate.is_file()):
        relative = file_path.relative_to(source_dir)
        destination = temp_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    temp_dir.replace(target_dir)
    target_hash = file_or_tree_hash(target_dir)
    return {
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "copied": True,
        "hash_verified": source_hash.get("sha256") == target_hash.get("sha256"),
        "source_hash": source_hash,
        "target_hash": target_hash,
    }


def format_kir(value: float) -> str:
    return f"{value:.2f}"


def quote_command(argv: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Write artifacts and print the launch plan without starting training.")
    parser.add_argument("--dataset", type=str, help="Canonical or official dataset name, e.g. stackoverflow or banking77.")
    parser.add_argument("--kir", type=float, help="Known intent ratio passed through as --known_cls_ratio.")
    parser.add_argument("--seed", type=int, help="Training seed.")
    parser.add_argument("--epochs", type=float, help="Epoch count for the official entrypoint.")
    parser.add_argument("--train-batch-size", type=int, help="Optional smoke-only training batch-size override.")
    parser.add_argument("--eval-batch-size", type=int, help="Optional smoke-only evaluation batch-size override.")
    parser.add_argument("--device", type=str, help="Execution device: cpu, cuda, or cuda:N.")
    parser.add_argument("--gpu-id", type=str, dest="gpu_id", help="Explicit GPU id override for the official entrypoint.")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing completed run directory instead of refusing to continue.")
    return parser.parse_args(argv)


def canonical_dataset_name(name: str, aliases: dict[str, str]) -> str:
    key = name.strip().lower()
    if key not in aliases:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported dataset {name!r}. Expected one of: {allowed}")
    return aliases[key]


def resolve_device(device_value: str | None, gpu_id: str | None, default_device: str, default_gpu_id: str) -> tuple[str, str]:
    raw = (device_value or default_device).strip().lower()
    if raw == "cpu":
        return "cpu", "-1"
    if raw == "cuda":
        return "cuda", gpu_id or default_gpu_id
    if raw.startswith("cuda:"):
        return "cuda", raw.split(":", 1)[1]
    raise ValueError(f"Unsupported device {device_value!r}. Expected cpu, cuda, or cuda:N")


def build_command(resolved: dict[str, Any], effective_gpu_id: str) -> list[str]:
    argv = [
        sys.executable,
        str(resolved["runtime_script"]),
        "--official-script",
        str(resolved["official_script"]),
        "--compat-root",
        str(resolved["compat_root"]),
        "--official-root",
        str(Path(resolved["official_script"]).parent),
        "--data_dir",
        str(resolved["launch_data_root"]),
        "--save_results_path",
        str(resolved["results_dir"]),
        "--pretrain_dir",
        str(resolved["pretrain_dir"]),
        "--bert_model",
        str(resolved["bert_model"]),
        "--dataset",
        resolved["official_dataset"],
        "--known_cls_ratio",
        str(resolved["kir"]),
        "--labeled_ratio",
        str(resolved["labeled_ratio"]),
        "--seed",
        str(resolved["seed"]),
        "--gpu_id",
        effective_gpu_id,
        "--num_train_epochs",
        str(resolved["epochs"]),
        "--train_batch_size",
        str(resolved["train_batch_size"]),
        "--eval_batch_size",
        str(resolved["eval_batch_size"]),
        "--wait_patient",
        str(resolved["wait_patient"]),
    ]
    if resolved["freeze_bert_parameters"]:
        argv.append("--freeze_bert_parameters")
    if resolved["save_results"]:
        argv.append("--save_results")
    return argv


def collect_attempt_history(run_dir: Path) -> list[dict[str, Any]]:
    logs_dir = run_dir / "logs"
    attempts: list[dict[str, Any]] = []
    if not logs_dir.is_dir():
        return attempts
    for stderr_path in sorted(logs_dir.glob("stderr-*.log")):
        entry = {
            "stderr_log": str(stderr_path),
            "stdout_log": str(stderr_path.with_name(stderr_path.name.replace("stderr-", "stdout-", 1))),
            "runtime_error_detected": False,
        }
        try:
            tail = stderr_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            tail = ""
        if "corrupted double-linked list" in tail:
            entry["runtime_error_detected"] = True
            entry["blocker"] = "native_allocator_corruption"
        elif "RuntimeError:" in tail:
            entry["runtime_error_detected"] = True
            if "modified by an inplace operation" in tail:
                entry["blocker"] = "stale_graph_inplace_version_mismatch"
        attempts.append(entry)
    return attempts


def ensure_launch_ready(resolved: dict[str, Any]) -> None:
    required = {
        "official_script": resolved["official_script"],
        "runtime_script": resolved["runtime_script"],
        "launch_data_root": resolved["launch_data_root"],
        "launch_dataset_dir": resolved["launch_dataset_dir"],
        "bert_model": resolved["bert_model"],
        "compat_root": resolved["compat_root"],
    }
    missing = [name for name, path in required.items() if not Path(path).exists()]
    if missing:
        names = ", ".join(sorted(missing))
        raise FileNotFoundError(f"Cannot launch MOGB smoke because required paths are missing: {names}")


def write_preflight_files(
    run_dir: Path,
    resolved_config: dict[str, Any],
    command_payload: dict[str, Any],
    environment_payload: dict[str, Any],
    source_hashes: dict[str, Any],
) -> None:
    atomic_write_json(run_dir / "resolved_config.json", resolved_config)
    atomic_write_json(run_dir / "command.json", command_payload)
    atomic_write_json(run_dir / "environment.json", environment_payload)
    atomic_write_json(run_dir / "source_hashes.json", source_hashes)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.resolve()
    config = load_config(config_path)

    aliases = {str(key).lower(): str(value) for key, value in (config.get("dataset_aliases") or {}).items()}
    official_dataset_names = {str(key): str(value) for key, value in (config.get("official_dataset_names") or {}).items()}
    defaults = dict(config.get("defaults") or {})
    metadata = dict(config.get("metadata") or {})

    dataset = canonical_dataset_name(args.dataset or str(config.get("default_dataset", "")), aliases)
    if dataset not in official_dataset_names:
        raise ValueError(f"Missing official dataset mapping for {dataset}")
    kir = float(args.kir if args.kir is not None else defaults.get("kir", 0.5))
    seed = int(args.seed if args.seed is not None else defaults.get("seed", 0))
    epochs = float(args.epochs if args.epochs is not None else defaults.get("epochs", 1.0))
    train_batch_size = int(args.train_batch_size if args.train_batch_size is not None else defaults.get("train_batch_size", 128))
    eval_batch_size = int(args.eval_batch_size if args.eval_batch_size is not None else defaults.get("eval_batch_size", 2048))
    device_mode, effective_gpu_id = resolve_device(
        args.device,
        args.gpu_id,
        str(defaults.get("device", "cuda")),
        str(defaults.get("gpu_id", "0")),
    )

    data_snapshot_root = resolve_repo_path(str(config["data_snapshot_root"]))
    official_script = resolve_repo_path(str(config["official_script"]))
    compat_root = resolve_repo_path(str(config["compat_root"]))
    runtime_script = resolve_repo_path(str(config["runtime_script"]))
    bert_model = resolve_repo_path(str(config["bert_model"]))
    output_root = resolve_repo_path(str(config["output_root"]))
    dataset_source_dir = (data_snapshot_root / dataset).resolve()
    if data_snapshot_root != DEFAULT_SOURCE_ROOT.resolve():
        raise ValueError(
            "This runner is pinned to data/sources/textoir/dffe2b1b848a069a6808f8089b4cb9bd16e2062b and must not read ../textoir."
        )

    run_dir = output_root / dataset / f"kir_{format_kir(kir)}" / f"seed_{seed}"
    logs_dir = run_dir / "logs"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stdout_log = logs_dir / f"stdout-{timestamp}.log"
    stderr_log = logs_dir / f"stderr-{timestamp}.log"
    manifest_path = run_dir / MANIFEST_NAME
    pretrain_dir = run_dir / "pretrain"
    results_dir = run_dir / "results"
    launch_data_root = run_dir / "data_snapshot"
    launch_dataset_dir = launch_data_root / official_dataset_names[dataset]

    if manifest_path.is_file() and not args.dry_run:
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise RuntimeError(f"Run directory already has a manifest; use --resume to reuse {run_dir}")
        if existing_manifest.get("status") == "complete":
            existing_manifest["resume_checked_at"] = now_iso()
            atomic_write_json(manifest_path, existing_manifest)
            print(json.dumps({"status": "complete", "run_dir": str(run_dir), "manifest": str(manifest_path)}))
            return 0

    resolved_config = {
        "config_path": str(config_path),
        "experiment_id": metadata.get("experiment_id", "mogb_official_modernized_smoke_v1"),
        "protocol_version": metadata.get("protocol_version"),
        "dataset": dataset,
        "official_dataset": official_dataset_names[dataset],
        "kir": kir,
        "seed": seed,
        "epochs": epochs,
        "device_mode": device_mode,
        "gpu_id": effective_gpu_id,
        "official_script": str(official_script),
        "runtime_script": str(runtime_script),
        "compat_root": str(compat_root),
        "data_snapshot_root": str(data_snapshot_root),
        "dataset_source_dir": str(dataset_source_dir),
        "launch_data_root": str(launch_data_root),
        "launch_dataset_dir": str(launch_dataset_dir),
        "bert_model": str(bert_model),
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "pretrain_dir": str(pretrain_dir),
        "results_dir": str(results_dir),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
        "labeled_ratio": float(defaults.get("labeled_ratio", 1.0)),
        "freeze_bert_parameters": bool(defaults.get("freeze_bert_parameters", True)),
        "save_results": bool(defaults.get("save_results", True)),
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "wait_patient": int(defaults.get("wait_patient", 10)),
        "representation": metadata.get("representation"),
        "partition": metadata.get("partition"),
        "boundary": metadata.get("boundary"),
        "distance": metadata.get("distance"),
        "runtime_repair": bool(metadata.get("runtime_repair", True)),
        "note": metadata.get(
            "note",
            "Separate smoke of pinned official logic under a compatibility layer; not a strict official reproduction.",
        ),
    }

    command_argv = build_command(resolved_config, effective_gpu_id)
    launch_data_root.mkdir(parents=True, exist_ok=True)
    copy_report = copy_dataset_snapshot(dataset_source_dir, launch_dataset_dir)
    matplotlib_cache = run_dir / "matplotlib_cache"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    py_path_entries = [str(compat_root), str(official_script.parent)]
    environment = {
        "cwd": str(S2C_ROOT),
        "python_executable": sys.executable,
        "pythonpath_entries": py_path_entries,
        "pythonpath": os.pathsep.join(py_path_entries),
        "cuda_visible_devices": "" if device_mode == "cpu" else effective_gpu_id,
        "device_mode": device_mode,
        "gpu_id": effective_gpu_id,
        "pythondontwritebytecode": "1",
        "mplconfigdir": str(matplotlib_cache),
    }
    command_payload = {
        "argv": command_argv,
        "shell_quoted": quote_command(command_argv),
        "cwd": environment["cwd"],
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }

    source_hashes = {
        "official_checkout": file_or_tree_hash(official_script.parent),
        "compat_checkout": file_or_tree_hash(compat_root),
        "data_snapshot_root": file_or_tree_hash(data_snapshot_root),
        "dataset_source_dir": file_or_tree_hash(dataset_source_dir),
        "artifact_dataset_copy": copy_report["target_hash"],
        "artifact_copy_report": copy_report,
        "bert_model": file_or_tree_hash(bert_model),
        "runner_script": file_or_tree_hash(SCRIPT_PATH),
        "runtime_script": file_or_tree_hash(runtime_script),
        "config_file": file_or_tree_hash(config_path),
    }
    launch_ready = all(
        bool(source_hashes[key]["exists"])
        for key in (
            "official_checkout",
            "data_snapshot_root",
            "dataset_source_dir",
            "bert_model",
            "compat_checkout",
            "runtime_script",
        )
    ) and bool(copy_report["hash_verified"])

    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_preflight_files(run_dir, resolved_config, command_payload, environment, source_hashes)
    attempt_history = collect_attempt_history(run_dir)
    known_blocker = next((entry for entry in attempt_history if entry.get("runtime_error_detected")), None)

    manifest = {
        "status": "dry_run" if args.dry_run else "prepared",
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
        "launch_ready": launch_ready,
        "strict_official_reproduction": False,
        "runtime_repair": bool(resolved_config["runtime_repair"]),
        "message": resolved_config["note"],
        "created_at": now_iso(),
        "run_dir": str(run_dir),
        "artifact_data_copy_verified": bool(copy_report["hash_verified"]),
        "artifact_data_copy_path": str(launch_dataset_dir),
        "resolved_config_path": str(run_dir / "resolved_config.json"),
        "command_path": str(run_dir / "command.json"),
        "environment_path": str(run_dir / "environment.json"),
        "source_hashes_path": str(run_dir / "source_hashes.json"),
        "stdout_log_path": str(stdout_log),
        "stderr_log_path": str(stderr_log),
        "dataset": dataset,
        "official_dataset": official_dataset_names[dataset],
        "kir": kir,
        "seed": seed,
        "epochs": epochs,
        "attempt_history": attempt_history,
        "known_blocker": known_blocker,
    }
    atomic_write_json(manifest_path, manifest)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "launch_ready": launch_ready,
                    "run_dir": str(run_dir),
                    "manifest": str(manifest_path),
                    "command": command_payload["shell_quoted"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    ensure_launch_ready(resolved_config)
    if not copy_report["hash_verified"]:
        raise RuntimeError(f"Artifact snapshot hash mismatch for {launch_dataset_dir}")
    launch_env = os.environ.copy()
    launch_env["PYTHONPATH"] = environment["pythonpath"]
    launch_env["CUDA_VISIBLE_DEVICES"] = environment["cuda_visible_devices"]
    launch_env["PYTHONDONTWRITEBYTECODE"] = environment["pythondontwritebytecode"]
    launch_env["PYTHONUNBUFFERED"] = "1"
    # The official code imports matplotlib during startup.  The workspace home
    # is read-only in some runners, so keep its cache inside this isolated run
    # directory instead of allowing an untracked global cache build to stall a
    # smoke cell.
    launch_env["MPLCONFIGDIR"] = environment["mplconfigdir"]

    started_at = now_iso()
    manifest.update({"status": "running", "started_at": started_at})
    atomic_write_json(manifest_path, manifest)
    try:
        with stdout_log.open("w", encoding="utf-8") as stdout_handle, stderr_log.open("w", encoding="utf-8") as stderr_handle:
            completed = subprocess.run(
                command_argv,
                cwd=S2C_ROOT,
                env=launch_env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
    except KeyboardInterrupt:
        manifest.update({"status": "interrupted", "completed_at": now_iso(), "returncode": 130})
        atomic_write_json(manifest_path, manifest)
        print(json.dumps({"status": manifest["status"], "run_dir": str(run_dir), "manifest": str(manifest_path), "returncode": 130}))
        return 130
    manifest.update(
        {
            "status": "complete" if completed.returncode == 0 else "failed",
            "completed_at": now_iso(),
            "returncode": completed.returncode,
        }
    )
    atomic_write_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "run_dir": str(run_dir), "manifest": str(manifest_path), "returncode": completed.returncode}))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
