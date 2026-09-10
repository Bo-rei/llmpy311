"""保存 TextOIR 原生方法的逐样本机制分析产物。

该模块由一次性 runtime overlay 调用，不修改 TextOIR 仓库。只保存数值数组和
方法/分数定义，不保存原始文本；它让后续的错误转移、分数排序和表示几何图有
可审计的输入，而不是从最终 F1 反推机制。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _save_arrays(output_dir: Path, arrays: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(output_dir / f"{name}.npy", value, allow_pickle=False)


def _save_manifest(output_dir: Path, payload: dict) -> None:
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _save_adb_analysis(args, data, method, outputs, output_dir: Path) -> None:
    features = []
    labels = []
    method.model.eval()
    with torch.no_grad():
        for batch in method.test_dataloader:
            batch = tuple(t.to(method.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            features.append(
                method.model(input_ids, segment_ids, input_mask, feature_ext=True)
                .detach()
                .cpu()
                .numpy()
            )
            labels.append(label_ids.detach().cpu().numpy())
    features = np.concatenate(features, axis=0).astype(np.float32)
    labels = np.concatenate(labels, axis=0).astype(np.int64)
    centroids = method.centroids.detach()
    deltas = method.delta.detach()
    distances = torch.cdist(torch.from_numpy(features).to(method.device), centroids)
    nearest_distance, nearest_label = distances.min(dim=1)
    radius = deltas[nearest_label]
    oos_score = nearest_distance / radius.clamp_min(1e-12)
    native_pred = nearest_label.clone()
    native_pred[oos_score >= 1.0] = data.unseen_label_id

    arrays = {
        "y_true": labels,
        "y_pred": np.asarray(outputs["y_pred"], dtype=np.int64),
        "native_pred": native_pred.cpu().numpy().astype(np.int64),
        "test_features": features,
        "nearest_label": nearest_label.cpu().numpy().astype(np.int64),
        "nearest_distance": nearest_distance.cpu().numpy().astype(np.float32),
        "radius": radius.cpu().numpy().astype(np.float32),
        "oos_score": oos_score.cpu().numpy().astype(np.float32),
    }
    _save_arrays(output_dir, arrays)
    _save_manifest(
        output_dir,
        {
            "schema_version": 1,
            "method": str(args.method),
            "score_definition": "nearest-centroid distance divided by learned radius; larger means more OOS",
            "sample_count": int(labels.size),
            "arrays": sorted(arrays),
            "raw_text_saved": False,
        },
    )


def save_analysis_artifacts(args, data, method, outputs) -> None:
    """保存 ADB/DA-ADB 的 native score 与 test representation。"""

    output_dir = Path(args.method_output_dir) / "analysis"
    method_name = str(args.method)
    if method_name in {"ADB", "DA-ADB"}:
        _save_adb_analysis(args, data, method, outputs, output_dir)
