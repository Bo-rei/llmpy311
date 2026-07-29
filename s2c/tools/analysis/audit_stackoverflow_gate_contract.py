"""Read-only StackOverflow Gate contract audit.

This command compares the historical ``nearest_sphere`` score with the
opt-in ``normalized_union`` contract on one frozen protocol_v2 cell.  It
reuses the E2 embedding cache, never encodes text, and writes only to a new
audit root.  The default detector mode is intentionally left unchanged so
this tool cannot mutate the E2/E3 historical contract by accident.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from protocol_v2.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from protocol_v2.data.hashing import atomic_write_json, sha256_file
from protocol_v2.evaluation.metrics import compute_binary_oos_metrics
from protocol_v2.experiments.mechanism_runner import load_e2_bundle
from protocol_v2.runtime.paths import ProtocolV2Paths


def _e2_run(paths: ProtocolV2Paths, dataset: str, seed: int, kir: float, k: int) -> Path:
    return (
        paths.run_root
        / "e2_gate_core_dense"
        / f"protocol_v2_textoir_v1__{dataset}__kir_{kir:.2f}__seed_{seed}__"
        f"repr_frozen_minilm__k_{k}__dist_mahalanobis_diag__boundary_mean_std"
    )


def _evaluate(bundle: Any, intents: np.ndarray, k: int, acceptance_mode: str) -> dict[str, Any]:
    detector = MultiSphereOOSDetector(
        radius_method="mean_std",
        radius_lambda=1.0,
        center_mode="class_centroid_mixture",
        distance_metric="mahalanobis_diag",
        covariance_eps=1e-6,
        l2_normalize=True,
        subcenters_per_intent=k,
        random_state=42,
        acceptance_mode=acceptance_mode,
    )
    detector.fit(bundle.train, intents)
    output = detector.predict_with_scores(bundle.test)
    labels = np.asarray([int(row["label"]) for row in bundle.views.test], dtype=np.int64)
    return {
        "k": k,
        "acceptance_mode": acceptance_mode,
        "sphere_count": len(detector.spheres),
        "mean_accepted_sphere_count": float(np.mean(output["accepted_sphere_count"])),
        **compute_binary_oos_metrics(labels, output["score"], threshold=1.0),
    }


def run_audit(
    paths: ProtocolV2Paths,
    *,
    dataset: str = "stackoverflow",
    seed: int = 42,
    kir: float = 0.50,
    output_root: Path | None = None,
) -> Path:
    if dataset != "stackoverflow":
        raise ValueError("This focused audit is intentionally restricted to StackOverflow")
    bundle = load_e2_bundle(paths, dataset, seed, kir)
    intents = np.asarray([str(row["intent"]) for row in bundle.views.train], dtype=object)
    rows: list[dict[str, Any]] = []
    for k in (1, 2):
        for mode in ("nearest_sphere", "normalized_union"):
            rows.append(_evaluate(bundle, intents, k, mode))

    source_runs = {}
    for k in (1, 2):
        run_dir = _e2_run(paths, dataset, seed, kir, k)
        manifest = run_dir / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Frozen E2 run is missing: {run_dir}")
        source_runs[str(k)] = {
            "run_id": run_dir.name,
            "manifest_sha256": sha256_file(manifest),
            "metrics_sha256": sha256_file(run_dir / "metrics.json"),
        }

    root = output_root or (paths.run_root / "detector_contract_audit_v1")
    root.mkdir(parents=True, exist_ok=True)
    output = root / f"{dataset}__kir_{kir:.2f}__seed_{seed}.json"
    atomic_write_json(
        output,
        {
            "schema_version": "s2c.detector_contract_audit.v1",
            "protocol_version": paths.dataset_version,
            "dataset": dataset,
            "kir": kir,
            "seed": seed,
            "uses_textoir_data": False,
            "encodes_new_embeddings": False,
            "source_e2_runs": source_runs,
            "results": rows,
            "note": (
                "nearest_sphere is the immutable historical E2 contract; "
                "normalized_union is an opt-in diagnostic and does not replace E2."
            ),
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--dataset", default="stackoverflow")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kir", type=float, default=0.50)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()
    paths = ProtocolV2Paths.discover(args.project_root)
    print(run_audit(paths, dataset=args.dataset, seed=args.seed, kir=args.kir, output_root=args.output_root))


if __name__ == "__main__":
    main()
