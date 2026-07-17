#!/usr/bin/env python3
"""导出 MiniLM 子簇的可解释 casebook。

这个脚本只做诊断，不把自动关键词或中心距离当作“语义有效”的证明。它为
人工审计准备同一套证据：每个 intent 的子簇大小、中心附近代表句、TF-IDF
关键词和簇间词汇 Jensen--Shannon 散度。人工评分字段保持为空，避免形成
“用 MiniLM 证明 MiniLM 簇有语义”的循环论证。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.experiments.cluster_separability.analysis import _json, _load_cache

V19_ROOT = PROJECT_ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cluster_separability_v19"
V21_ROOT = PROJECT_ROOT.parent / "artifacts" / "s2c" / "outputs" / "experiments" / "cluster_separability_v21"
DATASETS = ("clinc150", "banking77_oos", "stackoverflow")
SEEDS = (13, 42, 87)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _unit(v19_root: Path, dataset: str, seed: int, distance: str, k: int) -> Path:
    return v19_root / "fixed" / dataset / f"kir50_seed{seed}" / distance / f"k{k}"


def _load_inputs(v19_root: Path, dataset: str, seed: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    reference = _unit(v19_root, dataset, seed, "euclidean", 1)
    manifest = _json(reference / "run_manifest.json")
    data_root = Path(manifest["data_root"])
    rows = _json(data_root / "gate" / "train.json")
    embeddings = _load_cache(v19_root, dataset, 50, seed, "train", manifest)
    if len(rows) != len(embeddings):
        raise ValueError(f"train rows/cache mismatch: {dataset}/{seed}")
    return rows, np.asarray(embeddings, dtype=np.float64)


def _tokens(texts: list[str]) -> list[list[str]]:
    return [re.findall(r"[a-z][a-z0-9']+", text.lower()) for text in texts]


def _cluster_keywords(texts: list[str], top_n: int = 8) -> tuple[list[str], np.ndarray]:
    """返回簇级 TF-IDF 关键词和归一化词频分布。"""

    if not texts:
        return [], np.zeros(0, dtype=np.float64)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-z][a-z0-9']+\b",
        stop_words="english",
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # 极短或全为 stop-word 的簇仍应进入 casebook；关键词留空，不能让
        # 一个诊断性字段阻断整个实验矩阵。
        return [], np.zeros(0, dtype=np.float64)
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    terms = np.asarray(vectorizer.get_feature_names_out())
    order = np.argsort(-scores)[:top_n]
    # JSD 需要同一词表；这里返回簇内部词频，调用方会重新对齐词表。
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    counts = counts / max(float(counts.sum()), 1e-12)
    return [str(term) for term in terms[order]], counts


def _aligned_jsd(text_a: list[str], text_b: list[str]) -> float:
    token_a, token_b = _tokens(text_a), _tokens(text_b)
    vocabulary = sorted(set(token for row in token_a + token_b for token in row))
    if not vocabulary:
        return math.nan
    index = {token: i for i, token in enumerate(vocabulary)}
    distributions = []
    for rows in (token_a, token_b):
        counts = np.zeros(len(vocabulary), dtype=np.float64)
        for row in rows:
            for token in row:
                counts[index[token]] += 1.0
        total = float(counts.sum())
        if total <= 0:
            return math.nan
        counts /= total
        distributions.append(counts)
    if not np.isfinite(distributions[0]).all() or not np.isfinite(distributions[1]).all():
        return math.nan
    return float(jensenshannon(distributions[0], distributions[1], base=2.0) ** 2)


def _selected_k(v19_root: Path, dataset: str, seed: int, distance: str) -> int:
    table = pd.read_csv(v19_root / "selected_k_summary.csv")
    match = table[
        table["dataset"].eq(dataset)
        & table["kir"].eq(50)
        & table["data_seed"].eq(seed)
        & table["distance"].eq(distance)
    ]
    if len(match) != 1:
        raise ValueError(f"selected K is not unique: {dataset}/{seed}/{distance}")
    return int(match.iloc[0]["selected_k"])


def _case_rows(
    rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    dataset: str,
    seed: int,
    distance: str,
    k: int,
    top_n: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    x = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    intents = np.asarray([str(row["intent"]) for row in rows], dtype=object)
    case_rows: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []
    for intent in sorted(np.unique(intents).tolist()):
        indices = np.where(intents == intent)[0]
        points = x[indices]
        effective_k = min(int(k), len(points))
        if effective_k < 2:
            continue
        model = KMeans(n_clusters=effective_k, random_state=42, n_init=10)
        labels = model.fit_predict(points)
        centers = model.cluster_centers_
        clusters: dict[int, list[int]] = {cluster: np.where(labels == cluster)[0].tolist() for cluster in range(effective_k)}
        cluster_texts: dict[int, list[str]] = {
            cluster: [str(rows[indices[pos]]["text"]) for pos in positions]
            for cluster, positions in clusters.items()
        }
        center_separation = []
        for cluster in range(effective_k):
            other = [j for j in range(effective_k) if j != cluster]
            center_separation.append(float(np.min(1.0 - (centers[cluster] @ centers[other].T))))
        for cluster in range(effective_k):
            positions = clusters[cluster]
            # 按与中心的 cosine 距离取代表句，避免仅按文件顺序抽样。
            distances = 1.0 - points[positions] @ (centers[cluster] / max(np.linalg.norm(centers[cluster]), 1e-12))
            representative = [cluster_texts[cluster][i] for i in np.argsort(distances)[:top_n]]
            keywords, _ = _cluster_keywords(cluster_texts[cluster])
            jsd_values = [
                _aligned_jsd(cluster_texts[cluster], cluster_texts[other])
                for other in range(effective_k)
                if other != cluster
            ]
            case_rows.append(
                {
                    "dataset": dataset,
                    "kir": 50,
                    "data_seed": seed,
                    "distance": distance,
                    "intent": intent,
                    "requested_k": k,
                    "cluster": cluster,
                    "cluster_size": len(positions),
                    "cluster_ratio": len(positions) / len(indices),
                    "center_separation_min_cosine_distance": center_separation[cluster],
                    "keywords": "|".join(keywords),
                    "representative_texts": " || ".join(representative),
            "mean_pair_jsd": (
                float(np.nanmean(jsd_values))
                if jsd_values and np.isfinite(jsd_values).any()
                else math.nan
            ),
                }
            )
        if effective_k >= 2:
            annotation_rows.append(
                {
                    "dataset": dataset,
                    "kir": 50,
                    "data_seed": seed,
                    "distance": distance,
                    "intent": intent,
                    "requested_k": k,
                    "annotator_count": 0,
                    "semantic_submode_score_0_2": "",
                    "surface_only_score_0_2": "",
                    "annotator_notes": "",
                }
            )
    return case_rows, annotation_rows


def run_all(v19_root: Path = V19_ROOT, v21_root: Path = V21_ROOT) -> dict[str, Any]:
    case_rows: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []
    units = 0
    for dataset in DATASETS:
        for seed in SEEDS:
            rows, embeddings = _load_inputs(v19_root, dataset, seed)
            for distance in ("euclidean", "mahalanobis_diag"):
                ks = {2, _selected_k(v19_root, dataset, seed, distance)}
                for k in sorted(value for value in ks if value > 1):
                    cases, annotations = _case_rows(rows, embeddings, dataset=dataset, seed=seed, distance=distance, k=k)
                    case_rows.extend(cases)
                    annotation_rows.extend(annotations)
                    units += 1
    _write_csv(v21_root / "minilm_cluster_casebook.csv", case_rows)
    _write_csv(v21_root / "minilm_cluster_annotation_template.csv", annotation_rows)
    manifest = {
        "protocol": "minilm_cluster_semantic_casebook_v21",
        "source_root": str(v19_root),
        "output_root": str(v21_root),
        "v19_frozen": True,
        "units": units,
        "case_rows": len(case_rows),
        "annotation_rows": len(annotation_rows),
        "human_annotation_required": True,
        "automatic_keywords_are_diagnostic_only": True,
    }
    v21_root.mkdir(parents=True, exist_ok=True)
    (v21_root / "minilm_cluster_casebook_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export MiniLM cluster semantic casebook")
    parser.add_argument("--v19-root", type=Path, default=V19_ROOT)
    parser.add_argument("--v21-root", type=Path, default=V21_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(run_all(args.v19_root, args.v21_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
