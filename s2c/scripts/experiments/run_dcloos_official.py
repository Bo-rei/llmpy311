#!/usr/bin/env python3
"""Run one isolated DCLOOS end-to-end contract cell.

The upstream repository is copied into a run-local overlay.  Only compatibility
changes needed by the current Python/Transformers stack are applied there:
the legacy AdamW import, local BERT/tokenizer paths, a tiny TensorBoard logger
shim, and explicit metric/prediction serialization.  The pinned third-party
checkout and all protocol data remain read-only.

The external SQuAD corpus is accepted only as a caller-provided file.  The
official Drive snapshot is named ``squad.tsv`` while the upstream loader asks
for ``squad_placeh.tsv``; the byte-identical rename is recorded as a
reproduction assumption in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "third_party" / "dcloos_source"
DEFAULT_ARTIFACT_ROOT = ROOT.parent / "artifacts" / "s2c" / "external" / "dcloos_official_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for candidate in sorted(path.rglob("*")):
        if not candidate.is_file() or ".git" in candidate.parts or "__pycache__" in candidate.parts:
            continue
        digest.update(candidate.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(candidate)))
        digest.update(b"\0")
    return digest.hexdigest()


def git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def copy_file(source: Path, target: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_hash = sha256_file(source)
    target_hash = sha256_file(target)
    if source_hash != target_hash:
        raise RuntimeError(f"copy hash mismatch: {source} -> {target}")
    return {"source": str(source), "target": str(target), "sha256": source_hash, "size_bytes": target.stat().st_size}


def patch_text(path: Path, old: str, new: str, label: str, patches: list[dict[str, Any]]) -> None:
    before = path.read_text(encoding="utf-8")
    count = before.count(old)
    if count != 1:
        raise RuntimeError(f"expected one {label} in {path}, found {count}")
    after = before.replace(old, new)
    path.write_text(after, encoding="utf-8")
    patches.append({"file": str(path), "label": label, "source_sha256": hashlib.sha256(before.encode()).hexdigest(), "overlay_sha256": sha256_file(path)})


def build_overlay(root: Path) -> tuple[Path, list[dict[str, Any]]]:
    overlay = root / "runtime_overlay"
    if overlay.exists():
        shutil.rmtree(overlay)
    shutil.copytree(SOURCE_ROOT, overlay, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    patches: list[dict[str, Any]] = []
    patch_text(
        overlay / "utils.py",
        "from transformers import AdamW",
        "from torch.optim import AdamW",
        "torch-native AdamW compatibility",
        patches,
    )
    patch_text(
        overlay / "models" / "Encoder.py",
        "import torch.nn as nn\nimport torch\n",
        "import os\nimport torch.nn as nn\nimport torch\n",
        "import os for local BERT resolution",
        patches,
    )
    patch_text(
        overlay / "models" / "Encoder.py",
        "self.encoder = BertModel.from_pretrained('bert-base-uncased')",
        "self.encoder = BertModel.from_pretrained(os.environ.get('DCLOOS_BERT_MODEL', 'bert-base-uncased'))",
        "local BERT model path",
        patches,
    )
    patch_text(
        overlay / "dataloader.py",
        "tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)",
        "tokenizer = BertTokenizer.from_pretrained(os.environ.get('DCLOOS_BERT_MODEL', 'bert-base-uncased'), do_lower_case=True)",
        "local tokenizer path",
        patches,
    )
    patch_text(
        overlay / "dataloader.py",
        "tokens_a = tokenizer.encode_plus(",
        "tokens_a = tokenizer(",
        "Transformers tokenizer call compatibility",
        patches,
    )
    patch_text(
        overlay / "dataloader.py",
        "pad_to_max_length=True,",
        "padding='max_length',",
        "Transformers padding argument compatibility",
        patches,
    )
    main = overlay / "main.py"
    patch_text(
        main,
        "import os\nimport sys\n",
        "import os\nimport json\nimport sys\n",
        "json import for metric serialization",
        patches,
    )
    patch_text(
        main,
        "            f1_scores = f1_score(np.concatenate(labels, axis=0), np.concatenate(preds, axis=0), average=None)\n",
        "            labels_array = np.concatenate(labels, axis=0).reshape(-1)\n            preds_array = np.concatenate(preds, axis=0).reshape(-1)\n            prediction_path = os.environ.get('DCLOOS_PREDICTIONS_PATH')\n            if prediction_path:\n                np.savez_compressed(prediction_path, y_true=labels_array, y_pred=preds_array)\n            f1_scores = f1_score(labels_array, preds_array, average=None)\n",
        "explicit prediction serialization",
        patches,
    )
    patch_text(
        main,
        "    # save the last model\n    # save_file = os.path.join(\n    #     opt.save_folder, 'last.pth')\n    # save_model(model, optimizer_bert, opt, opt.epochs, save_file)\n",
        "    metrics_path = os.environ.get('DCLOOS_METRICS_PATH')\n    if metrics_path:\n        # The upstream loop only evaluates test data when validation improves.\n        # A run can therefore finish without defining test-local variables;\n        # perform one final test pass so metrics/predictions are always emitted.\n        if 'f1_scores' not in locals():\n            test_acc, test_oos_acc, f1_scores = evaluation(opt, model, dataset, mode='test')\n        metrics = {\n            'accuracy': float(test_acc.item()),\n            'oos_recall': float(test_oos_acc.item()),\n            'f1_all': float(np.mean(f1_scores) * 100.0),\n            'f1_u': float(f1_scores[-1] * 100.0),\n            'f1_k': float(np.mean(f1_scores[:-1]) * 100.0),\n            'best_epoch': int(epoch),\n            'known_class_count': int(dataset.num_labels),\n            'test_sample_count': int(len(dataset.test_examples)),\n        }\n        with open(metrics_path, 'w', encoding='utf-8') as metric_handle:\n            json.dump(metrics, metric_handle, indent=2, sort_keys=True)\n            metric_handle.write('\\n')\n\n    # save the last model\n    # save_file = os.path.join(\n    #     opt.save_folder, 'last.pth')\n    # save_model(model, optimizer_bert, opt, opt.epochs, save_file)\n",
        "explicit metric serialization",
        patches,
    )
    (overlay / "tensorboard_logger.py").write_text(
        """class Logger:\n    def __init__(self, logdir=None, flush_secs=2):\n        self.logdir = logdir\n    def log_value(self, *args, **kwargs):\n        return None\n""",
        encoding="utf-8",
    )
    patches.append({"file": str(overlay / "tensorboard_logger.py"), "label": "minimal TensorBoard logger shim", "overlay_sha256": sha256_file(overlay / "tensorboard_logger.py")})
    return overlay, patches


def positive_snapshot(source: Path, runtime_root: Path, dataset: str) -> list[dict[str, Any]]:
    copies = []
    target = runtime_root / dataset
    for split in ("train", "dev", "test"):
        copies.append(copy_file(source / f"{split}.tsv", target / f"{split}.tsv"))
    return copies


def compute_metrics(raw: dict[str, Any], predictions: Path) -> dict[str, Any]:
    import numpy as np

    arrays = np.load(predictions)
    y_true = arrays["y_true"].astype(int)
    y_pred = arrays["y_pred"].astype(int)
    oos_id = int(raw["known_class_count"])
    true_oos = y_true == oos_id
    pred_oos = y_pred == oos_id
    tp = int(np.sum(true_oos & pred_oos))
    fp = int(np.sum(~true_oos & pred_oos))
    fn = int(np.sum(true_oos & ~pred_oos))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    known_mask = ~true_oos
    known_recall = float(np.mean(y_pred[known_mask] != oos_id)) if np.any(known_mask) else 0.0
    return raw | {
        "oos_precision": precision * 100.0,
        "oos_recall": recall * 100.0,
        "oos_f1": f1 * 100.0,
        "known_recall": known_recall * 100.0,
        "known_to_oos": int(np.sum(known_mask & pred_oos)),
        "oos_to_known": int(np.sum(true_oos & ~pred_oos)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not shutil.which(args.python):
        raise FileNotFoundError(args.python)
    if not Path(args.positive_dir).is_dir():
        raise FileNotFoundError(args.positive_dir)
    if not Path(args.negative_tsv).is_file():
        raise FileNotFoundError(args.negative_tsv)
    if not torch_cuda_available():
        raise RuntimeError("DCLOOS official run requires CUDA; CPU fallback is not a faithful contract")
    artifact = Path(args.output_dir).resolve()
    if artifact.exists() and any(artifact.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {artifact}")
    artifact.mkdir(parents=True, exist_ok=True)
    runtime_data = artifact / "runtime_data"
    source_data = artifact / "source_snapshot"
    if runtime_data.exists():
        shutil.rmtree(runtime_data)
    if source_data.exists():
        shutil.rmtree(source_data)
    source_data.mkdir(parents=True)
    runtime_data.mkdir(parents=True)
    positive = positive_snapshot(Path(args.positive_dir), runtime_data, args.dataset_pos)
    negative_source = Path(args.negative_tsv)
    negative_copy = copy_file(negative_source, source_data / "squad.tsv")
    negative_runtime = copy_file(negative_source, runtime_data / "squad" / "squad_placeh.tsv")
    overlay, patches = build_overlay(artifact)
    model_path = Path(args.bert_model).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    metrics_path = artifact / "raw_metrics.json"
    predictions_path = artifact / "predictions.npz"
    command = [
        args.python,
        str(overlay / "main.py"),
        "--data_dir", str(runtime_data),
        "--dataset_pos", args.dataset_pos,
        "--dataset_neg", "squad",
        "--dl_large",
        "--loss_ce_only",
        "--know_only",
        "--known_cls_ratio", str(args.known_cls_ratio),
        "--train_batch_size", str(args.train_batch_size),
        "--eval_batch_size", str(args.eval_batch_size),
        "--n_oos", str(args.n_oos),
        "--num_convex", str(args.num_convex),
        "--num_convex_val", str(args.num_convex_val),
        "--temp", str(args.temp),
        "--patient", str(args.patient),
        "--seed", str(args.seed),
        "--lr", str(args.lr),
        "--num_train_epochs", str(args.max_epochs),
        "--save_results_path", str(artifact / "source_results"),
        "--datetime", args.datetime,
    ]
    env = os.environ.copy()
    env.update({
        "DCLOOS_BERT_MODEL": str(model_path),
        "DCLOOS_METRICS_PATH": str(metrics_path),
        "DCLOOS_PREDICTIONS_PATH": str(predictions_path),
        "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
    })
    try:
        completed = subprocess.run(command, cwd=overlay, env=env, capture_output=True, text=True, check=False)
    except KeyboardInterrupt:
        atomic_json(
            artifact / "run_manifest.json",
            {
                "status": "timeout_incomplete",
                "experiment_id": args.experiment_id,
                "dataset_pos": args.dataset_pos,
                "dataset_neg": "squad",
                "known_cls_ratio": args.known_cls_ratio,
                "seed": args.seed,
                "command": command,
                "official_source_repo": "https://github.com/fanolabs/out-of-scope-intent-detection",
                "official_source_commit": git(SOURCE_ROOT, "rev-parse", "HEAD"),
                "negative_corpus_sha256": sha256_file(negative_source),
                "overlay_patch": patches,
                "intermediate_predictions_excluded": True,
                "note": "Process interrupted at the declared runtime ceiling; no final metrics were produced.",
            },
        )
        raise
    (artifact / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (artifact / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        atomic_json(artifact / "run_manifest.json", {"status": "failed", "return_code": completed.returncode, "command": command, "patches": patches})
        raise RuntimeError(f"DCLOOS failed with return code {completed.returncode}; see {artifact / 'stderr.log'}")
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise RuntimeError("DCLOOS exited successfully without metrics/predictions")
    raw = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = compute_metrics(raw, predictions_path)
    atomic_json(artifact / "metrics.json", metrics)
    manifest = {
        "status": "complete",
        "experiment_id": args.experiment_id,
        "dataset_pos": args.dataset_pos,
        "dataset_neg": "squad",
        "known_cls_ratio": args.known_cls_ratio,
        "seed": args.seed,
        "official_source_repo": "https://github.com/fanolabs/out-of-scope-intent-detection",
        "official_source_commit": git(SOURCE_ROOT, "rev-parse", "HEAD"),
        "official_redirect_repo": "https://github.com/liam0949/DCLOOS",
        "positive_snapshot": positive,
        "negative_snapshot": {"raw": negative_copy, "runtime_renamed_copy": negative_runtime, "rename_assumption": "official Drive exposes squad.tsv; upstream dataloader requires squad_placeh.tsv; bytes are unchanged"},
        "negative_corpus_sha256": sha256_file(negative_source),
        "overlay_patch": patches,
        "overlay_tree_sha256": tree_hash(overlay),
        "bert_model": {"path": str(model_path), "tree_sha256": tree_hash(model_path)},
        "command": command,
        "return_code": completed.returncode,
        "metrics_sha256": sha256_file(artifact / "metrics.json"),
        "predictions_sha256": sha256_file(predictions_path),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_json(artifact / "run_manifest.json", manifest)
    return {"status": "complete", "artifact_root": str(artifact), "metrics": metrics}


def torch_cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-pos", default="oos")
    parser.add_argument("--positive-dir", type=Path, required=True)
    parser.add_argument("--negative-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--bert-model", type=Path, default=ROOT.parent / "assets" / "models" / "bert-base-uncased")
    parser.add_argument("--known-cls-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--max-epochs", type=int, default=1000)
    parser.add_argument("--patient", type=int, default=100)
    parser.add_argument("--train-batch-size", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=100)
    parser.add_argument("--n-oos", type=int, default=200)
    parser.add_argument("--num-convex", type=int, default=400)
    parser.add_argument("--num-convex-val", type=int, default=200)
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--datetime", default="20210401")
    parser.add_argument("--experiment-id", default="dcloos_official_single_cell_v1")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
