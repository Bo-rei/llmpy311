#!/usr/bin/env python3
"""在独立子进程和一次性 runtime overlay 中运行上游 TextOIR 方法。

设计边界是“上游仓库始终只读”：每个 run 拷贝一份 detection 代码，只在拷贝中
修正已失效的 BERT 路径和 BERT-only 运行所需的 optional-import 兼容问题。每个改动
都保存源/目标 SHA256，并用 allowlist 拒绝任何意外修改。s2c 不直接 import
TextOIR，从而不把其旧依赖带入当前 Python 环境。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from ._common import (
        DATASETS,
        METHOD_CONTRACTS,
        SPLITS,
        benchmark_labels,
        default_textoir_root,
        git_output,
        sha256_file,
        write_json,
    )
except ImportError:  # Direct script execution.
    from _common import (
        DATASETS,
        METHOD_CONTRACTS,
        SPLITS,
        benchmark_labels,
        default_textoir_root,
        git_output,
        sha256_file,
        write_json,
    )


METHODS = METHOD_CONTRACTS


def probe_python_environment(python_executable: str, gpu_id: str) -> dict:
    """在目标解释器中验证 TextOIR 运行依赖和指定 CUDA 设备。

    环境探针刻意放在独立子进程中。这样即使旧版 torch/CUDA 组合发生原生崩溃，
    也不会拖垮负责写 manifest 的 s2c 进程；同时真正分配一个 CUDA tensor，避免
    只依赖版本字符串或 ``nvidia-smi`` 得出误判。
    """

    probe = r"""
import importlib.metadata
import json
import sys

import easydict
import numpy
import pandas
import scipy
import sklearn
import torch
import transformers

gpu_id = int(sys.argv[1])
device_count = torch.cuda.device_count()
if gpu_id < 0 or gpu_id >= device_count:
    raise RuntimeError(f"CUDA device {gpu_id} is unavailable; device_count={device_count}")
tensor = torch.zeros(1, device=f"cuda:{gpu_id}")
distribution_versions = {}
for name in ("torch", "transformers", "scikit-learn", "numpy", "scipy", "pandas", "easydict"):
    try:
        distribution_versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        distribution_versions[name] = None
packages = {
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "scikit-learn": sklearn.__version__,
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "pandas": pandas.__version__,
    "easydict": distribution_versions["easydict"],
}
print(json.dumps({
    "python_executable": sys.executable,
    "python_version": sys.version,
    "packages": packages,
    "distribution_versions": distribution_versions,
    "package_files": {
        "torch": torch.__file__,
        "transformers": transformers.__file__,
        "scikit-learn": sklearn.__file__,
        "numpy": numpy.__file__,
        "scipy": scipy.__file__,
        "pandas": pandas.__file__,
        "easydict": easydict.__file__,
    },
    "torch_cuda_version": torch.version.cuda,
    "cuda_device_count": device_count,
    "selected_gpu_id": gpu_id,
    "selected_gpu_name": torch.cuda.get_device_name(gpu_id),
    "selected_gpu_capability": list(torch.cuda.get_device_capability(gpu_id)),
    "torch_compiled_arch_list": torch.cuda.get_arch_list(),
    "cuda_allocation_verified": tensor.device.type == "cuda",
}, sort_keys=True))
"""
    completed = subprocess.run(
        [python_executable, "-c", probe, gpu_id],
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"TEXTOIR environment probe failed for {python_executable}: {detail}"
        )
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"TEXTOIR environment probe returned invalid JSON: {completed.stdout!r}"
        ) from exc
    if result["packages"].get("easydict") is None:
        raise RuntimeError("TEXTOIR environment is missing the required easydict package")
    return result


def audit_run_artifacts(run_dir: Path) -> dict:
    """检查上游进程是否真的产生了可导入预测，而不只相信返回码。"""

    output_root = run_dir / "textoir_outputs" / "open_intent_detection"
    prediction_dirs = sorted(
        path
        for path in output_root.glob("*")
        if (path / "y_true.npy").is_file() and (path / "y_pred.npy").is_file()
    )
    result_csv = run_dir / "results" / "results.csv"
    complete = (
        len(prediction_dirs) == 1
        and (prediction_dirs[0] / "y_true.npy").stat().st_size > 0
        and (prediction_dirs[0] / "y_pred.npy").stat().st_size > 0
        and result_csv.is_file()
        and result_csv.stat().st_size > 0
    )
    payload = {
        "complete": complete,
        "prediction_directory_count": len(prediction_dirs),
        "prediction_directories": [str(path.resolve()) for path in prediction_dirs],
        "results_csv": str(result_csv.resolve()),
        "results_csv_exists": result_csv.is_file() and result_csv.stat().st_size > 0,
    }
    if len(prediction_dirs) == 1:
        payload["y_true_sha256"] = sha256_file(prediction_dirs[0] / "y_true.npy")
        payload["y_pred_sha256"] = sha256_file(prediction_dirs[0] / "y_pred.npy")
    return payload


def probe_method_runtime(
    python_executable: str,
    detection_root: Path,
    method: str,
) -> dict:
    """在训练前验证上游 registry、Transformers API 与方法原生扩展。"""

    probe = r"""
import json
import os
import sys

detection_root, method = sys.argv[1:3]
os.chdir(detection_root)
sys.path.insert(0, detection_root)
from transformers import AdamW
from backbones import backbones_map
from methods import method_map
if method not in method_map:
    raise RuntimeError(f"method {method!r} is absent from method_map")
openmax_libmr = None
if method == "OpenMax":
    from methods.OpenMax import openmax_utils
    if not hasattr(openmax_utils, "libmr") or not hasattr(openmax_utils.libmr, "MR"):
        raise RuntimeError("OpenMax requires a Python-ABI-compatible libMR.MR extension")
    openmax_libmr = True
print(json.dumps({
    "method": method,
    "registered": method in method_map,
    "backbones": sorted(backbones_map),
    "transformers_adamw_import": AdamW.__name__,
    "openmax_libmr": openmax_libmr,
}, sort_keys=True))
"""
    completed = subprocess.run(
        [python_executable, "-c", probe, str(detection_root), method],
        cwd=detection_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    payload = {
        "complete": completed.returncode == 0,
        "return_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode == 0:
        try:
            payload["details"] = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            payload["complete"] = False
            payload["parse_error"] = "preflight did not end with JSON"
    return payload


def select_known_labels(textoir_root: Path, dataset: str, ratio: float, seed: int) -> list[str]:
    """精确复制 TextOIR 基于 NumPy RandomState 的 known-label 抽样顺序。

    这是 TextOIR 第二协议，不会强行对齐 s2c 按 domain 平衡的 known-intent 选择。
    两套协议结果必须分表报告。
    """

    labels = benchmark_labels(textoir_root, dataset)
    count = round(len(labels) * ratio)
    state = np.random.RandomState(seed)
    return state.choice(np.asarray(labels), count, replace=False).tolist()


def require_clean_worktree(status_porcelain: str, *, dry_run: bool) -> None:
    """真实运行要求上游工作树干净；dry-run 仍允许用于排查问题。"""
    if status_porcelain and not dry_run:
        raise ValueError("Refusing to run against a dirty TEXTOIR worktree")


def resolve_bert_model(value: Path | None, *, dry_run: bool) -> Path | None:
    """解析本地 BERT 目录；真实运行禁止回退到上游失效的绝对路径。"""
    if value is None:
        if dry_run:
            return None
        raise ValueError("--bert-model is required for a non-dry-run TEXTOIR execution")
    path = value.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Local BERT model directory not found: {path}")
    return path


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def prepare_runtime_overlay(
    textoir_root: Path,
    run_dir: Path,
    config_name: str,
    bert_model: Path,
) -> tuple[Path, dict]:
    """构建一次性 detection overlay，并返回可审计的修改 provenance。

    允许变更仅有五类：当前 method 的 ``bert_model``、两处 Llama 顶层注册、
    upstream main 遗漏的 BERT dataloader 路由，以及 ADB 的 CUDA 诊断轨迹
    持久化。它们只恢复 BERT-only 官方命令的可执行性，不改动被选方法的
    训练损失、网络结构或数据协议。
    """

    source_root = textoir_root / "open_intent_detection"
    overlay_root = run_dir / "runtime_overlay" / "open_intent_detection"
    if overlay_root.exists():
        raise FileExistsError(f"Runtime overlay already exists: {overlay_root}")
    shutil.copytree(
        source_root,
        overlay_root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    relative_config = Path("configs") / f"{config_name}.py"
    source_config = source_root / relative_config
    overlay_config = overlay_root / relative_config
    source_text = source_config.read_text(encoding="utf-8")
    pattern = re.compile(
        r"(?P<prefix>['\"]bert_model['\"]\s*:\s*)(?P<quote>['\"])(?P<value>.*?)(?P=quote)"
    )
    patched_text, replacements = pattern.subn(
        lambda match: f"{match.group('prefix')}{str(bert_model)!r}",
        source_text,
    )
    if replacements != 1:
        raise ValueError(
            f"Expected one bert_model assignment in {source_config}, found {replacements}"
        )
    overlay_config.write_text(patched_text, encoding="utf-8")

    # 上游 __init__.py 会急切 import 所有可选 Llama 方法。即使本次选择 MSP，
    # 缺少 peft 仍会在参数解析前崩溃。这里只从 overlay 删除相关顶层注册。
    compatibility_specs = {
        Path("backbones/__init__.py"): (
            "from .llama import LLAMA_lora_Disaware",
            "'llama_disaware': LLAMA_lora_Disaware,",
        ),
        Path("methods/__init__.py"): (
            "from .ADB_llama.manager import ADBManager_llama",
            "'DA-ADB_llama' : ADBManager_llama,",
        ),
    }
    compatibility_patches = []
    for relative_path, removed_lines in compatibility_specs.items():
        source_path = source_root / relative_path
        overlay_path = overlay_root / relative_path
        lines = overlay_path.read_text(encoding="utf-8").splitlines(keepends=True)
        stripped = [line.strip() for line in lines]
        for expected in removed_lines:
            if stripped.count(expected) != 1:
                raise ValueError(
                    f"Expected one compatibility line {expected!r} in {source_path}"
                )
        overlay_path.write_text(
            "".join(line for line in lines if line.strip() not in removed_lines),
            encoding="utf-8",
        )
        compatibility_patches.append(
            {
                "file": relative_path.as_posix(),
                "reason": "avoid eager optional Llama/peft imports for a BERT-only run",
                "removed_lines": list(removed_lines),
                "source_sha256": sha256_file(source_path),
                "overlay_sha256": sha256_file(overlay_path),
            }
        )

    # 当前 upstream main 只把 ``bert_con`` 注册到 dataloader map，导致 README
    # 中 MSP/DOC/ADB 等官方命令全部在 DataManager 初始化时 KeyError。所有 BERT
    # backbone 原本就共用同一个 BERT_Loader；overlay 仅恢复这层缺失的路由表。
    dataloader_relative = Path("dataloaders/__init__.py")
    dataloader_source = source_root / dataloader_relative
    dataloader_overlay = overlay_root / dataloader_relative
    dataloader_text = dataloader_overlay.read_text(encoding="utf-8")
    dataloader_map = """backbone_loader_map = {
    'bert': BERT_Loader,
    'bert_norm': BERT_Loader,
    'bert_K+1-way': BERT_Loader,
    'bert_seg': BERT_Loader,
    'bert_disaware': BERT_Loader,
    'bert_doc': BERT_Loader,
    'bert_mdf': BERT_Loader,
    'bert_mdf_pretrain': BERT_Loader,
    'bert_knncl': BERT_Loader,
    'bert_con': BERT_Loader,
}"""
    patched_dataloader, dataloader_replacements = re.subn(
        r"backbone_loader_map\s*=\s*\{.*?\}",
        dataloader_map,
        dataloader_text,
        count=1,
        flags=re.DOTALL,
    )
    if dataloader_replacements != 1:
        raise ValueError(
            f"Expected one backbone_loader_map in {dataloader_source}, "
            f"found {dataloader_replacements}"
        )
    dataloader_overlay.write_text(patched_dataloader, encoding="utf-8")
    compatibility_patches.append(
        {
            "file": dataloader_relative.as_posix(),
            "reason": "restore official BERT backbone routes missing from upstream main",
            "registered_backbones": [
                "bert",
                "bert_norm",
                "bert_K+1-way",
                "bert_seg",
                "bert_disaware",
                "bert_doc",
                "bert_mdf",
                "bert_mdf_pretrain",
                "bert_knncl",
                "bert_con",
            ],
            "source_sha256": sha256_file(dataloader_source),
            "overlay_sha256": sha256_file(dataloader_overlay),
        }
    )

    method_compatibility_files: list[str] = []
    if config_name in {"ADB", "DA-ADB"}:
        # 官方 ADB 命令启用 --save_model，但 upstream 会把 CUDA tensor list
        # 直接交给 np.save，导致训练完成后、测试前崩溃。这里只修复诊断轨迹的
        # device 转换；用于决策的 best delta/centroid 和训练过程均不改变。
        adb_relative = Path("methods/ADB/manager.py")
        adb_source = source_root / adb_relative
        adb_overlay = overlay_root / adb_relative
        adb_text = adb_overlay.read_text(encoding="utf-8")
        old_save = (
            "np.save(os.path.join(args.method_output_dir, 'all_deltas.npy'), "
            "self.delta_points)"
        )
        new_save = (
            "np.save(os.path.join(args.method_output_dir, 'all_deltas.npy'), "
            "np.asarray([point.detach().cpu().numpy() for point in self.delta_points]))"
        )
        if adb_text.count(old_save) != 1:
            raise ValueError(f"Expected one all_deltas np.save call in {adb_source}")
        adb_overlay.write_text(adb_text.replace(old_save, new_save), encoding="utf-8")
        method_compatibility_files.append(adb_relative.as_posix())
        compatibility_patches.append(
            {
                "file": adb_relative.as_posix(),
                "reason": "move ADB diagnostic delta history to CPU before np.save",
                "source_sha256": sha256_file(adb_source),
                "overlay_sha256": sha256_file(adb_overlay),
            }
        )

    source_files = {
        path.relative_to(source_root).as_posix(): sha256_file(path)
        for path in source_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    overlay_files = {
        path.relative_to(overlay_root).as_posix(): sha256_file(path)
        for path in overlay_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    if source_files.keys() != overlay_files.keys():
        raise RuntimeError("Runtime overlay file set differs from the source repository")
    changed_files = sorted(
        relative for relative in source_files if source_files[relative] != overlay_files[relative]
    )
    compatibility_files = sorted(
        [path.as_posix() for path in compatibility_specs]
        + [dataloader_relative.as_posix()]
        + method_compatibility_files
    )
    expected_changed_files = sorted([relative_config.as_posix(), *compatibility_files])
    # allowlist 是兼容层的安全边界：如果 copy/patch 意外改动了其他文件，
    # 立即终止，而不带着未记录的语义修改训练。
    if changed_files != expected_changed_files:
        raise RuntimeError(f"Unexpected runtime overlay changes: {changed_files}")

    provenance = {
        "source_detection_root": str(source_root.resolve()),
        "overlay_detection_root": str(overlay_root.resolve()),
        "source_tree_sha256": _tree_hash(source_root),
        "overlay_tree_sha256": _tree_hash(overlay_root),
        "patched_config": relative_config.as_posix(),
        "source_config_sha256": source_files[relative_config.as_posix()],
        "overlay_config_sha256": overlay_files[relative_config.as_posix()],
        "changed_files": changed_files,
        "config_changed_files": [relative_config.as_posix()],
        "compatibility_changed_files": compatibility_files,
        "compatibility_patches": compatibility_patches,
        "bert_model": str(bert_model),
    }
    return overlay_root, provenance


def build_command(
    args: argparse.Namespace,
    run_dir: Path,
    detection_root: Path,
) -> list[str]:
    """将显式方法契约转换为上游 CLI，所有输出定向到当前 run_dir。"""

    method = METHODS[args.method]
    command = [
        args.python_executable,
        str(detection_root / "run.py"),
        "--dataset", args.dataset,
        "--method", args.method,
        "--known_cls_ratio", str(args.known_cls_ratio),
        "--labeled_ratio", "1.0",
        "--seed", str(args.seed),
        "--backbone", method["backbone"],
        "--config_file_name", method["config"],
        "--loss_fct", method["loss"],
        "--gpu_id", args.gpu_id,
        "--data_dir", str((args.textoir_root / "data").resolve()),
        "--output_dir", str((run_dir / "textoir_outputs").resolve()),
        "--log_dir", str((run_dir / "logs").resolve()),
        "--result_dir", str((run_dir / "results").resolve()),
        "--results_file_name", "results.csv",
        "--train",
        "--save_results",
    ]
    if method.get("pretrain"):
        command.append("--pretrain")
    if method.get("save_model"):
        command.append("--save_model")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textoir-root", type=Path, default=default_textoir_root())
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument(
        "--output-root",
        type=Path,
        help="Root under which dataset/method/KIR/seed directories are created",
    )
    destination.add_argument(
        "--run-dir",
        type=Path,
        help="Exact directory for one attempt; used by the resumable matrix runner",
    )
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=tuple(METHODS), required=True)
    parser.add_argument("--known-cls-ratio", type=float, choices=(0.25, 0.5, 0.75), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument(
        "--bert-model",
        type=Path,
        help="Local pretrained BERT directory; required unless --dry-run is used",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout in seconds")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """创建 manifest，执行隔离子进程，并在结束后复查 clone/overlay 状态。"""

    args = parse_args()
    args.textoir_root = args.textoir_root.resolve()
    if not (args.textoir_root / "open_intent_detection" / "run.py").is_file():
        raise FileNotFoundError(f"Invalid TEXTOIR repository: {args.textoir_root}")
    upstream_status = git_output(args.textoir_root, "status", "--porcelain")
    require_clean_worktree(upstream_status, dry_run=args.dry_run)
    bert_model = resolve_bert_model(args.bert_model, dry_run=args.dry_run)

    run_dir = (
        args.run_dir.resolve()
        if args.run_dir is not None
        else (
            args.output_root
            / args.dataset
            / args.method
            / f"kir{int(args.known_cls_ratio * 100)}"
            / f"seed{args.seed}"
        ).resolve()
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        detection_root = run_dir / "runtime_overlay" / "open_intent_detection"
        source_config = (
            args.textoir_root
            / "open_intent_detection"
            / "configs"
            / f"{METHODS[args.method]['config']}.py"
        )
        overlay_provenance = {
            "status": "planned",
            "source_detection_root": str(
                (args.textoir_root / "open_intent_detection").resolve()
            ),
            "overlay_detection_root": str(detection_root),
            "patched_config": f"configs/{METHODS[args.method]['config']}.py",
            "source_config_sha256": sha256_file(source_config),
            "bert_model": str(bert_model) if bert_model is not None else None,
        }
    else:
        assert bert_model is not None
        detection_root, overlay_provenance = prepare_runtime_overlay(
            args.textoir_root,
            run_dir,
            METHODS[args.method]["config"],
            bert_model,
        )
    command = build_command(args, run_dir, detection_root)
    known_labels = select_known_labels(
        args.textoir_root, args.dataset, args.known_cls_ratio, args.seed
    )
    environment_provenance = None
    method_preflight = None
    if not args.dry_run:
        environment_provenance = probe_python_environment(
            args.python_executable, args.gpu_id
        )
        method_preflight = probe_method_runtime(
            args.python_executable, detection_root, args.method
        )
    # manifest 在启动子进程前先落盘为 running，因此超时或崩溃也会
    # 留下 commit、known-label list、split hash 和完整命令，不会成为无 provenance 的失败目录。
    manifest = {
        "schema_version": 2,
        "status": "dry_run" if args.dry_run else "running",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runner_pid": os.getpid(),
        "textoir_root": str(args.textoir_root),
        "upstream_commit": git_output(args.textoir_root, "rev-parse", "HEAD"),
        "upstream_remote": git_output(args.textoir_root, "remote", "get-url", "origin"),
        "upstream_clean_before_run": not bool(upstream_status),
        "bert_model": str(bert_model) if bert_model is not None else None,
        "runtime_overlay": overlay_provenance,
        "dataset": args.dataset,
        "method": args.method,
        "known_cls_ratio": args.known_cls_ratio,
        "labeled_ratio": 1.0,
        "seed": args.seed,
        "known_labels": known_labels,
        "unknown_label": "oos" if args.dataset == "oos" else "<UNK>",
        "unknown_label_id": len(known_labels),
        "split_sha256": {
            split: sha256_file(args.textoir_root / "data" / args.dataset / f"{split}.tsv")
            for split in SPLITS
        },
        "command": command,
        "working_directory": str(detection_root),
        "environment": environment_provenance,
        "method_preflight": method_preflight,
    }
    write_json(run_dir / "run_manifest.json", manifest)
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    if not method_preflight["complete"]:
        manifest["status"] = "preflight_failed"
        manifest["return_code"] = 4
        manifest["upstream_clean_after_run"] = not bool(
            git_output(args.textoir_root, "status", "--porcelain")
        )
        manifest["runtime_overlay"]["overlay_tree_sha256_after_run"] = _tree_hash(
            detection_root
        )
        manifest["runtime_overlay"]["overlay_unchanged_after_run"] = (
            manifest["runtime_overlay"]["overlay_tree_sha256_after_run"]
            == manifest["runtime_overlay"]["overlay_tree_sha256"]
        )
        manifest["artifact_audit"] = audit_run_artifacts(run_dir)
        manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(run_dir / "run_manifest.json", manifest)
        (run_dir / "external_process.log").write_text(
            method_preflight["stdout"] + method_preflight["stderr"], encoding="utf-8"
        )
        return 4

    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(args.seed)
    # 不在 overlay 内生成 pyc，否则运行后 tree hash 会受解释器缓存干扰。
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    log_path = run_dir / "external_process.log"
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command,
                cwd=detection_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            return_code = 124
            log.write("\n[s2c compat] TEXTOIR subprocess timed out.\n")
    artifact_audit = audit_run_artifacts(run_dir)
    manifest["return_code"] = return_code
    manifest["timed_out"] = timed_out
    manifest["artifact_audit"] = artifact_audit
    manifest["status"] = (
        "complete"
        if return_code == 0 and artifact_audit["complete"]
        else ("timed_out" if timed_out else "failed")
    )
    manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    # 同时证明两个边界：上游 clone 仍干净，且 runtime overlay 的
    # 源码在运行期间未被上游脚本自修改。
    manifest["upstream_clean_after_run"] = not bool(
        git_output(args.textoir_root, "status", "--porcelain")
    )
    manifest["runtime_overlay"]["overlay_tree_sha256_after_run"] = _tree_hash(
        detection_root
    )
    manifest["runtime_overlay"]["overlay_unchanged_after_run"] = (
        manifest["runtime_overlay"]["overlay_tree_sha256_after_run"]
        == manifest["runtime_overlay"]["overlay_tree_sha256"]
    )
    write_json(run_dir / "run_manifest.json", manifest)
    print(run_dir)
    if manifest["status"] == "complete":
        return 0
    return return_code if return_code != 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
