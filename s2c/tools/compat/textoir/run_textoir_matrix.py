#!/usr/bin/env python3
"""顺序执行并可恢复地审计 TextOIR 第二协议实验矩阵。

每个实验单元使用不可变 ``attempt_XXXX`` 目录。失败或中断后的 ``--resume``
会创建新 attempt，而不是覆盖日志和 provenance；已完成且已经由 s2c 重新导入
预测的单元会直接跳过。GPU 训练保持串行，避免多个 BERT 任务争抢同一张卡。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterable

try:
    from ._common import DATASETS, METHOD_CONTRACTS, write_json
except ImportError:  # Direct script execution.
    from _common import DATASETS, METHOD_CONTRACTS, write_json


FIRST_BATCH_METHODS = ("MSP", "DOC", "ADB", "OpenMax")
METHODS = METHOD_CONTRACTS
KIRS = (0.25, 0.5, 0.75)
SEEDS = (0, 1, 2)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_matrix(
    datasets: Iterable[str],
    methods: Iterable[str],
    ratios: Iterable[float],
    seeds: Iterable[int],
) -> list[dict]:
    """固定排序生成矩阵，保证多次 resume 的单元顺序一致。"""

    return [
        {"dataset": dataset, "method": method, "known_cls_ratio": ratio, "seed": seed}
        for dataset in datasets
        for method in methods
        for ratio in ratios
        for seed in seeds
    ]


def unit_directory(output_root: Path, unit: dict) -> Path:
    return (
        output_root
        / unit["dataset"]
        / unit["method"]
        / f"kir{int(unit['known_cls_ratio'] * 100)}"
        / f"seed{unit['seed']}"
    )


def audit_attempt(attempt_dir: Path, expected_unit: dict | None = None) -> dict:
    """验证 attempt 的上游产物、clone 边界和 s2c 导入结果。"""

    manifest_path = attempt_dir / "run_manifest.json"
    imported_path = attempt_dir / "imported" / "import_summary.json"
    predictions_path = attempt_dir / "imported" / "predictions.jsonl"
    reasons: list[str] = []
    manifest = None
    imported = None
    if not manifest_path.is_file():
        reasons.append("missing run_manifest.json")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            reasons.append("invalid run_manifest.json")
        if manifest is not None:
            if manifest.get("status") != "complete":
                reasons.append(f"run status is {manifest.get('status')!r}")
            if manifest.get("return_code") != 0:
                reasons.append("non-zero upstream return code")
            if not manifest.get("upstream_clean_after_run"):
                reasons.append("upstream clone was not clean after run")
            if not manifest.get("runtime_overlay", {}).get("overlay_unchanged_after_run"):
                reasons.append("runtime overlay changed during run")
            if not manifest.get("artifact_audit", {}).get("complete"):
                reasons.append("upstream prediction artifacts are incomplete")
            if expected_unit is not None:
                for key in ("dataset", "method", "known_cls_ratio", "seed"):
                    if manifest.get(key) != expected_unit[key]:
                        reasons.append(f"run manifest {key} does not match matrix unit")
    if not imported_path.is_file():
        reasons.append("missing imported/import_summary.json")
    else:
        try:
            imported = json.loads(imported_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            reasons.append("invalid imported/import_summary.json")
        if imported is not None and expected_unit is not None:
            for key in ("dataset", "method", "known_cls_ratio", "seed"):
                if imported.get(key) != expected_unit[key]:
                    reasons.append(f"import summary {key} does not match matrix unit")
    if not predictions_path.is_file() or predictions_path.stat().st_size == 0:
        reasons.append("missing imported/predictions.jsonl")
    return {
        "attempt_dir": str(attempt_dir.resolve()),
        "complete": not reasons,
        "reasons": reasons,
        "run_status": manifest.get("status") if manifest else None,
    }


def attempt_directories(unit_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in (unit_dir / "attempts").glob("attempt_[0-9][0-9][0-9][0-9]")
        if path.is_dir()
    )


def next_attempt_directory(unit_dir: Path) -> Path:
    attempts = attempt_directories(unit_dir)
    number = int(attempts[-1].name.rsplit("_", 1)[-1]) + 1 if attempts else 1
    return unit_dir / "attempts" / f"attempt_{number:04d}"


def completed_attempt(unit_dir: Path, expected_unit: dict | None = None) -> Path | None:
    for attempt in reversed(attempt_directories(unit_dir)):
        if audit_attempt(attempt, expected_unit)["complete"]:
            return attempt
    return None


def matrix_status(output_root: Path, matrix: list[dict]) -> dict:
    rows = []
    for unit in matrix:
        directory = unit_directory(output_root, unit)
        attempts = attempt_directories(directory)
        complete = completed_attempt(directory, unit)
        rows.append(
            {
                **unit,
                "unit_directory": str(directory.resolve()),
                "attempts": len(attempts),
                "latest_attempt": str(attempts[-1].resolve()) if attempts else None,
                "complete_attempt": str(complete.resolve()) if complete else None,
                "status": "complete" if complete else ("attempted" if attempts else "missing"),
            }
        )
    missing = [row for row in rows if row["status"] != "complete"]
    return {
        "schema_version": 1,
        "expected_units": len(rows),
        "complete_units": len(rows) - len(missing),
        "missing_units": len(missing),
        "rows": rows,
        "missing": missing,
    }


def discover_completed_runs(output_root: Path) -> list[tuple[dict, Path]]:
    """发现输出根目录下的全部成功单元，而非仅查看本次命令的子矩阵。

    TextOIR 的训练按方法串行执行，实际使用时经常以 ``--methods ADB`` 之类的
    小批次续跑。若汇总只遍历当前命令的矩阵，后一次续跑会把先前 MSP/DOC 的
    成功行从共享 CSV 中抹掉。这里以目录协议还原单元，并再次执行完整性审计；
    因此失败 attempt 仍被保留用于诊断，但绝不会进入论文汇总。
    """

    discovered: list[tuple[dict, Path]] = []
    for unit_dir in sorted(output_root.glob("*/*/kir[0-9]*/seed[0-9]*")):
        if not unit_dir.is_dir():
            continue
        try:
            relative = unit_dir.relative_to(output_root)
            dataset, method, kir_part, seed_part = relative.parts
            unit = {
                "dataset": dataset,
                "method": method,
                "known_cls_ratio": int(kir_part.removeprefix("kir")) / 100,
                "seed": int(seed_part.removeprefix("seed")),
            }
        except (ValueError, TypeError):
            # 非协议目录可能由人工分析产生；汇总器应忽略它，而不是阻断续跑。
            continue
        attempt = completed_attempt(unit_dir, unit)
        if attempt is not None:
            discovered.append((unit, attempt))
    return discovered


def export_metric_summaries(output_root: Path) -> None:
    """从全部已完成 attempt 重建逐 seed 指标与同协议 mean/std。"""

    metric_names = ("accuracy", "known_macro_f1", "open_oos_f1", "macro_f1")
    raw_rows = []
    for unit, attempt in discover_completed_runs(output_root):
        imported = json.loads(
            (attempt / "imported" / "import_summary.json").read_text(encoding="utf-8")
        )
        raw_rows.append(
            {
                **unit,
                **{name: imported["metrics"][name] for name in metric_names},
                "attempt_dir": str(attempt.resolve()),
            }
        )

    raw_path = output_root / "textoir_results_by_seed.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["dataset", "method", "known_cls_ratio", "seed", *metric_names, "attempt_dir"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    groups: dict[tuple[str, str, float], list[dict]] = {}
    for row in raw_rows:
        groups.setdefault(
            (row["dataset"], row["method"], row["known_cls_ratio"]), []
        ).append(row)
    summary_rows = []
    for (dataset, method, ratio), rows in sorted(groups.items()):
        summary = {
            "dataset": dataset,
            "method": method,
            "known_cls_ratio": ratio,
            "seeds": len(rows),
        }
        for name in metric_names:
            values = [float(row[name]) for row in rows]
            summary[f"{name}_mean"] = statistics.fmean(values)
            # 与常见三种子实验汇总一致，std 使用总体定义（ddof=0）。
            summary[f"{name}_std"] = statistics.pstdev(values)
        summary_rows.append(summary)
    summary_path = output_root / "textoir_baseline_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["dataset", "method", "known_cls_ratio", "seeds"] + [
            f"{name}_{suffix}" for name in metric_names for suffix in ("mean", "std")
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)


def build_external_command(args: argparse.Namespace, unit: dict, attempt_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "tools.compat.textoir.run_external_textoir",
        "--textoir-root",
        str(args.textoir_root.resolve()),
        "--run-dir",
        str(attempt_dir.resolve()),
        "--dataset",
        unit["dataset"],
        "--method",
        unit["method"],
        "--known-cls-ratio",
        str(unit["known_cls_ratio"]),
        "--seed",
        str(unit["seed"]),
        "--gpu-id",
        args.gpu_id,
        "--bert-model",
        str(args.bert_model.resolve()),
        "--python-executable",
        # venv 的 python 通常是指向 base interpreter 的符号链接；这里不能
        # ``resolve()``，否则会绕过 venv 自己安装的 easydict/site-packages。
        str(args.python_executable.absolute()),
    ]
    if args.timeout is not None:
        command.extend(["--timeout", str(args.timeout)])
    return command


def run_unit(args: argparse.Namespace, unit: dict) -> dict:
    directory = unit_directory(args.output_root, unit)
    complete = completed_attempt(directory, unit)
    if complete is not None:
        if not args.resume:
            raise FileExistsError(
                f"Completed unit already exists; use --resume to skip it: {directory}"
            )
        return {"status": "skipped", "attempt_dir": str(complete.resolve())}
    attempts = attempt_directories(directory)
    if attempts and not args.resume:
        raise FileExistsError(
            f"Incomplete attempts already exist; use --resume to preserve and retry: {directory}"
        )

    attempt_dir = next_attempt_directory(directory)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    command = build_external_command(args, unit, attempt_dir)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    import_completed = None
    if completed.returncode == 0:
        import_completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.compat.textoir.import_textoir_results",
                "--run-dir",
                str(attempt_dir.resolve()),
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    orchestration = {
        "schema_version": 1,
        "unit": unit,
        "attempt_dir": str(attempt_dir.resolve()),
        "command": command,
        "started_at_utc": started,
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "external_return_code": completed.returncode,
        "external_stdout": completed.stdout,
        "external_stderr": completed.stderr,
        "import_return_code": import_completed.returncode if import_completed else None,
        "import_stdout": import_completed.stdout if import_completed else None,
        "import_stderr": import_completed.stderr if import_completed else None,
    }
    write_json(attempt_dir / "orchestration.json", orchestration)
    audit = audit_attempt(attempt_dir, unit)
    return {"status": "complete" if audit["complete"] else "failed", **audit}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textoir-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bert-model", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument(
        "--methods", nargs="+", choices=tuple(METHODS), default=list(FIRST_BATCH_METHODS)
    )
    parser.add_argument(
        "--known-cls-ratios", nargs="+", type=float, choices=KIRS, default=list(KIRS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit and summarize existing attempts without starting training",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = build_matrix(args.datasets, args.methods, args.known_cls_ratios, args.seeds)
    if args.dry_run:
        print(json.dumps({"units": len(matrix), "matrix": matrix}, indent=2, sort_keys=True))
        return 0

    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        status = matrix_status(args.output_root, matrix)
        write_json(args.output_root / "matrix_status.json", status)
        export_metric_summaries(args.output_root)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0 if status["missing_units"] == 0 else 2
    manifest = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "matrix": matrix,
        "textoir_root": str(args.textoir_root.resolve()),
        "bert_model": str(args.bert_model.resolve()),
        "python_executable": str(args.python_executable.absolute()),
        "gpu_id": args.gpu_id,
    }
    write_json(args.output_root / "matrix_manifest.json", manifest)

    for index, unit in enumerate(matrix, start=1):
        result = run_unit(args, unit)
        print(f"[{index}/{len(matrix)}] {unit}: {result['status']}", flush=True)
        write_json(args.output_root / "matrix_status.json", matrix_status(args.output_root, matrix))
        export_metric_summaries(args.output_root)

    status = matrix_status(args.output_root, matrix)
    manifest["status"] = "complete" if status["missing_units"] == 0 else "partial"
    manifest["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["complete_units"] = status["complete_units"]
    manifest["missing_units"] = status["missing_units"]
    write_json(args.output_root / "matrix_manifest.json", manifest)
    write_json(args.output_root / "matrix_status.json", status)
    export_metric_summaries(args.output_root)
    return 0 if status["missing_units"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
