"""Train the matched SmolLM Router/Expert components for aligned Banking77."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART_ROOT = ROOT.parent / "artifacts" / "s2c" / "analysis" / "banking_textoir_aligned"
DATA_ROOT = ART_ROOT / "data"
COMPONENTS_ROOT = ART_ROOT / "components"
MODEL_ROOT = ROOT.parent / "assets" / "models"
KIRS = (0.25, 0.50, 0.75)
SEEDS = (13, 42, 87)


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kirs", nargs="+", type=float, default=list(KIRS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    env = dict(
        os.environ,
        CONDA_DEFAULT_ENV="bo",
        PYTHONUNBUFFERED="1",
        PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "src"))),
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )
    manifest_path = ART_ROOT / "components_manifest.json"
    manifest = {
        "status": "running",
        "experiment": "banking77_textoir_aligned_components",
        "dataset": "banking77",
        "dataset_variant": "standard_77_intent",
        "protocol": "protocol_v2_textoir_v1",
        "data_root": str(DATA_ROOT),
        "components_root": str(COMPONENTS_ROOT),
        "kirs": list(args.kirs),
        "seeds": list(args.seeds),
        "router": "single-domain constant router; no router training required",
        "expert": "SmolLM-135M + LoRA, Known train/dev only, test deferred",
        "test_used_for_selection": False,
        "real_oos_used_for_training": False,
        "real_oos_used_for_selection": False,
        "pseudo_oos_used": False,
        "device": "cuda",
    }
    dump(manifest_path, manifest)
    for kir in args.kirs:
        for seed in args.seeds:
            tag = f"kir{round(kir * 100):02d}_seed{seed}"
            data = DATA_ROOT / "banking77" / tag
            out = COMPONENTS_ROOT / "banking77" / tag
            complete = out / "selection_complete.json"
            if complete.is_file():
                continue
            domains = sorted(path.name for path in (data / "experts").iterdir() if path.is_dir())
            if domains != ["banking"]:
                raise ValueError(f"unexpected aligned Banking77 domains: {domains}")
            experts_root = out / "experts"
            router_path = out / "router" / "constant_router.json"
            dump(router_path, {"router_mode": "constant", "domain": "banking", "domain_label": 0})
            for domain in domains:
                domain_checkpoint = experts_root / domain / "best_model.pt"
                if domain_checkpoint.is_file():
                    continue
                command = [
                    sys.executable,
                    str(ROOT / "tools" / "train" / "train_expert_v19.py"),
                    "--domain", domain,
                    "--model_path", str(MODEL_ROOT / "smollm135m"),
                    "--data_dir", str(data / "experts"),
                    "--output_dir", str(experts_root),
                    "--epochs", "15",
                    "--batch_size", "32",
                    "--patience", "5",
                    "--num_workers", "0",
                    "--seed", str(seed),
                    "--defer_test",
                ]
                log_path = out / "training.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            dump(
                complete,
                {
                    "dataset": "banking77",
                    "dataset_variant": "standard_77_intent",
                    "kir": kir,
                    "seed": seed,
                    "router": str(router_path),
                    "experts": str(experts_root),
                    "domains": domains,
                    "device": "cuda",
                    "test_read": False,
                    "test_used_for_selection": False,
                    "real_oos_used_for_training": False,
                    "real_oos_used_for_selection": False,
                    "pseudo_oos_used": False,
                },
            )
            print(f"COMPONENTS banking77/{tag}", flush=True)
    manifest.update(status="complete")
    dump(manifest_path, manifest)
    print("BANKING_TEXTOIR_ALIGNED_COMPONENTS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
