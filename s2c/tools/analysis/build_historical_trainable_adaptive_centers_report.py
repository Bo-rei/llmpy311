#!/usr/bin/env python3
"""Report frozen adaptive-center experiment decisions and aggregate evidence."""
from __future__ import annotations

import csv
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def read(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def historical_reconciliation(replay_pipeline=False):
    from tools.legacy.analysis_v19.run_trainable_minilm_historical_v1 import _row_id
    from tools.eval.run_historical_trainable_full_pipeline import (
        _load_rows, _make_pipeline, _detector_state, DEFAULT_H1_ROOT, compute_metrics,
    )
    import torch
    records = []
    for full in read(ROOT / "results/analysis/historical_trainable_full_pipeline/per_seed.csv"):
        dataset, seed = full["dataset"], int(full["seed"])
        source = DEFAULT_H1_ROOT / dataset / f"kir50_seed{seed}" / "trainable_k1"
        saved = json.loads((source / "metrics.json").read_text())
        with (source / "predictions.jsonl").open() as handle:
            samples = [json.loads(line) for line in handle if line.strip()]
        rows = _load_rows(dataset, seed)
        assert len(rows) == len(samples)
        assert all(r["intent"] == s["gold_intent"] and int(r["label"]) == int(s["gold_is_oos"])
                   for r, s in zip(rows, samples, strict=True))
        assert all(_row_id(r, "test", i) == s["sample_id"] for i, (r, s) in enumerate(zip(rows, samples, strict=True)))
        truth = np.array([int(s["gold_is_oos"]) for s in samples])
        pred = np.array([int(s["predicted_is_oos"]) for s in samples])
        gold = ["__oos__" if s["gold_is_oos"] else s["gold_intent"] for s in samples]
        guessed = ["__oos__" if s["predicted_is_oos"] else s["predicted_intent"] for s in samples]
        known_labels = sorted({r["intent"] for r in rows if int(r["label"]) == 0})
        gate_f1 = f1_score(truth, pred, zero_division=0)
        gate_macro = f1_score(gold, guessed, labels=known_labels + ["__oos__"], average="macro", zero_division=0)
        gate_acc = float(np.mean(np.array(gold) == np.array(guessed)))
        np.testing.assert_allclose([gate_f1, gate_macro, gate_acc],
                                   [saved["oos_f1"], saved["f1_all"], saved["accuracy"]], atol=1e-10, rtol=0)
        stages = {k.removeprefix("stage_count_"): int(v) for k, v in full.items() if k.startswith("stage_count_")}
        measured = {k: float(full[k]) for k in ("oos_f1", "f1_all", "overall_accuracy", "known_recall", "false_accept_rate", "router_error_rate", "expert_error_rate")}
        transitions = {}
        if replay_pipeline:
            signature = json.loads((source / "detector_signature.json").read_text())
            with tempfile.TemporaryDirectory(prefix="s2c_metric_replay_") as temporary:
                detector = Path(temporary) / "detector.json"
                detector.write_text(json.dumps(_detector_state(signature)), encoding="utf-8")
                pipeline = _make_pipeline(dataset, seed, torch.device("cuda"), detector, DEFAULT_H1_ROOT)
                predictions = pipeline.predict_batch([r["text"] for r in rows], batch_size=128)
                measured, stages = compute_metrics(rows, predictions)
                np.testing.assert_array_equal(pred, [int(p["is_oos"]) for p in predictions])
                gate_correct = np.array(gold) == np.array(guessed)
                pipeline_correct = np.array(gold) == np.array([
                    "__oos__" if p["is_oos"] else p.get("intent", "__missing__") for p in predictions])
                repaired = int(np.sum(~gate_correct & pipeline_correct))
                regressed = int(np.sum(gate_correct & ~pipeline_correct))
                np.testing.assert_allclose(measured["overall_accuracy"] - gate_acc,
                                           (repaired - regressed) / len(rows), atol=1e-10, rtol=0)
                transitions = {"downstream_repaired_count": repaired, "downstream_regressed_count": regressed,
                               "oos_decision_mismatch_count": 0}
                del pipeline, predictions
                torch.cuda.empty_cache()
        records.append({"dataset": dataset, "seed": seed, "kir": .5,
            "gate_oos_f1_recomputed": float(gate_f1), "gate_oos_f1_saved": saved["oos_f1"],
            "full_pipeline_oos_f1": measured["oos_f1"],
            "gate_oos_recall": float(np.mean(pred[truth == 1] == 1)),
            "gate_macro_f1": float(gate_macro), "full_pipeline_macro_f1": measured["f1_all"],
            "gate_accuracy": gate_acc, "full_pipeline_accuracy": measured["overall_accuracy"],
            "oos_f1_delta": measured["oos_f1"] - gate_f1,
            "macro_f1_delta": measured["f1_all"] - gate_macro,
            "accuracy_delta": measured["overall_accuracy"] - gate_acc,
            "gate_known_recall": float(np.mean(pred[truth == 0] == 0)),
            "pipeline_known_recall": measured["known_recall"],
            "gate_false_acceptance": float(np.mean(pred[truth == 1] == 0)),
            "pipeline_false_acceptance": measured["false_accept_rate"],
            "router_error_rate": measured["router_error_rate"], "expert_error_rate": measured["expert_error_rate"],
            **{f"stage_count_{k}": v for k, v in stages.items()},
            **transitions,
            "historical_saved_pipeline_macro_f1": float(full["f1_all"]),
            "historical_saved_pipeline_accuracy": float(full["overall_accuracy"]),
            "replay_minus_saved_pipeline_macro_f1": measured["f1_all"] - float(full["f1_all"]),
            "replay_minus_saved_pipeline_accuracy": measured["overall_accuracy"] - float(full["overall_accuracy"]),
            "pipeline_metrics_evidence": "fresh_sample_level_cuda_replay" if replay_pipeline else "saved_aggregate",
            "gate_metrics_source": str(source / "metrics.json"), "semantic_gate_enabled": False})
        print(f"RECONCILED {dataset}/{seed}: macro delta={100*(measured['f1_all']-gate_macro):+.2f}pp; "
              f"replay-saved macro={100*(measured['f1_all']-float(full['f1_all'])):+.4f}pp", flush=True)
    return records


def build(replay_pipeline=False):
    result = ROOT / "results/analysis/historical_trainable_adaptive_centers"
    manifest = json.loads((result / "MANIFEST.json").read_text())
    selected = read(result / "selected_test_per_seed.csv")
    summaries = read(result / "selected_test_summary.csv")
    means = {(r["dataset"], r["selection_scope"]): float(r["oos_f1_mean"]) for r in summaries}
    adaptive_comparison = [f"- {dataset}：{100*(means[dataset, 'adaptive']-means[dataset, 'fixed']):+.2f} pp。"
                           for dataset in manifest["datasets"]]
    for summary in summaries:
        group = [r for r in selected if r["dataset"] == summary["dataset"] and r["selection_scope"] == summary["selection_scope"]]
        summary["direct_pipeline_verified"] = bool(group) and all(str(r["direct_pipeline_verified"]).lower() == "true" for r in group)
    write(result / "selected_test_summary.csv", summaries)
    locks, diagnostics, errors, strategies = [], [], [], []
    for dataset in manifest["datasets"]:
        for seed in manifest["seeds"]:
            cell = result / dataset / f"seed{seed}"
            locks.extend(json.loads((cell / "selection_lock.json").read_text())["choices"])
            diagnostics.extend(read(cell / "ranking_diagnostics.csv"))
            errors.extend(read(cell / "intent_errors.csv"))
            val = read(cell / "validation_workpoints.csv")
            test = read(cell / "test_workpoints.csv")
            lookup = {(r["strategy"], r["radius_lambda"], r["acceptance_mode"], r["threshold"]): r for r in test}
            for strategy in sorted({r["strategy"] for r in val}):
                candidates = [r for r in val if r["strategy"] == strategy]
                v = min(candidates, key=lambda r: (-float(r["oos_f1"]), int(r["center_count"]),
                    abs(float(r["radius_lambda"]) - 1), abs(float(r["threshold"]) - 1),
                    ("nearest_sphere", "normalized_union").index(r["acceptance_mode"])))
                t = lookup[(v["strategy"], v["radius_lambda"], v["acceptance_mode"], v["threshold"])]
                strategies.append({**t, "validation_oos_f1": v["oos_f1"], "test_used_for_selection": False})
    write(result / "selected_validation_per_seed.csv", locks)
    write(result / "ranking_diagnostics.csv", diagnostics)
    write(result / "intent_errors.csv", errors)
    write(result / "strategy_validation_selected_test.csv", strategies)
    reconciliation_path = result / "historical_gate_pipeline_metric_reconciliation.csv"
    reconciliation = read(reconciliation_path) if reconciliation_path.exists() else []
    if replay_pipeline or not reconciliation or not all(
        r.get("pipeline_metrics_evidence") == "fresh_sample_level_cuda_replay" for r in reconciliation
    ):
        reconciliation = historical_reconciliation(replay_pipeline)
        write(reconciliation_path, reconciliation)
    lines = ["# H1 自适应 per-intent 中心数量实验", "",
             "所有主结果由 validation OOS F1 选择，test 只确认；Known F1、Accuracy、Known Recall 均无硬约束。",
             "本实验属于 historical_v19_paper_main 的 H1 controlled evidence，论文 Ours 仅作历史数值参照。", "",
             "## 三 seed 测试结果", "",
             "| 数据集 | 选择范围 | OOS F1 均值±std | 相对论文 pp | 超过论文 seed 数 | 真实 pipeline |",
             "|---|---|---:|---:|---:|---|"]
    for r in summaries:
        lines.append(f"| {r['dataset']} | {r['selection_scope']} | {float(r['oos_f1_mean'])*100:.2f}±{float(r['oos_f1_std'])*100:.2f} | {float(r['delta_oos_f1_pp']):+.2f} | {r['seeds_beating_paper']}/{r['seed_count']} | {r['direct_pipeline_verified']} |")
    lines.extend(["", "std 使用 ddof=0，与历史报告一致；三 seed 不代表统计显著性。overall 从统一和自适应候选共同选择，fixed/adaptive 是各自 validation 最优。", "",
                  "## 逐 seed 锁定配置与辅助指标", "",
                  "| 数据集/seed | 策略 | λ / threshold / mode | Val OOS | Test OOS | Known Recall | False acceptance | Pipeline macro F1 / Accuracy |",
                  "|---|---|---|---:|---:|---:|---:|---:|"])
    for r in selected:
        if r["selection_scope"] != "overall":
            continue
        aux = "pending"
        if "full_pipeline_f1_all" in r:
            aux = f"{float(r['full_pipeline_f1_all'])*100:.2f} / {float(r['full_pipeline_overall_accuracy'])*100:.2f}"
        lines.append(f"| {r['dataset']}/{r['seed']} | {r['strategy']} | {r['radius_lambda']} / {r['threshold']} / {r['acceptance_mode']} | {float(r['validation_oos_f1'])*100:.2f} | {float(r['oos_f1'])*100:.2f} | {float(r['known_recall'])*100:.2f} | {float(r['false_accept_rate'])*100:.2f} | {aux} |")
    lines.extend(["", "## 中心数、边界与 seed 工作点", "",
                  "对所有 27 个唯一策略完成同一粗网格；top100% 拆分策略与 fixed K=2/3 完全相同，以 alias 登记而不重复计算。",
                  "排序仅用归一化 Known train embedding：平均欧氏距离、总类内方差、K=2 相对 K=1 的 SSE 相对下降。拆分 intent 数向上取整，排序并列按 intent 名。",
                  "每个策略的 validation 粗网格最优 λ/mode 固定后，在其 threshold±.04 内以 .01 细化，并限制在 [.70,1.30]。所有配置、选择和中心映射先保存，再开始 test 指标计算。",
                  "Known F1/Accuracy 搜索表为 Gate 最近 intent 分类指标；full_pipeline_* 为真实下游指标，两者不能混用。", "",
                  "固定 K 与自适应 K 各自在 validation 上选参后的 test 差值（adaptive−fixed）：",
                  *adaptive_comparison, "",
                  "所以不能说自适应 K 普遍优于固定 K；CLINC/Banking 有小幅收益，SO 的 adaptive 子集均值反而较低。overall 是 validation 在两类候选间的选择，不根据 test 在 fixed/adaptive 间改选；其 test 均值也不保证高于两个子集。", "",
                  "## CLINC score ranking 与 intent 错误诊断", "",
                  "| Seed | Split | 选定 OOS F1 | 固定 score 最优阈值 oracle | AUROC | 全拒绝 OOS F1 |",
                  "|---|---|---:|---:|---:|---:|"])
    for r in diagnostics:
        if r["dataset"] == "clinc150":
            lines.append(f"| {r['seed']} | {r['split']} | {float(r['selected_oos_f1'])*100:.2f} | {float(r['fixed_score_threshold_oracle_f1'])*100:.2f} | {float(r['auroc'])*100:.2f} | {float(r['all_reject_oos_f1'])*100:.2f} |")
    lines.extend(["", "oracle 是已选 score 的事后阈值上限诊断，不用于选择或作为泛化结果，也不是所有可能表示的理论上限。near-boundary 定义为 |score/threshold−1|≤.05，只作边界难度代理，不能等同人工语义 Near-OOS 标签。", "",
                  "| CLINC test OOS intent | 误接收/样本数（三 seed 汇总） | 边界附近样本数 |", "|---|---:|---:|"])
    aggregate = {}
    for r in errors:
        if r["dataset"] == "clinc150" and r["split"] == "test" and r["is_known"] == "False":
            a = aggregate.setdefault(r["intent"], [0, 0, 0])
            for i, field in enumerate(("accepted_count", "count", "near_boundary_count")):
                a[i] += int(r[field])
    for intent, values in sorted(aggregate.items(), key=lambda pair: -pair[1][0])[:12]:
        lines.append(f"| {intent} | {values[0]}/{values[1]} | {values[2]} |")
    lines.extend(["", "## 原始 K=1 Gate 与 full pipeline 逐样本对账", "",
                  "以下为历史未调边界的 Trainable checkpoint，不是上方 validation-selected 自适应配置。数值均为百分数，差值为百分点。",
                  "Gate 分类采用最近中心 intent；pipeline 分类采用 Router/Expert。两者在相同全体测试样本和 Known intents+OOS 标签集合上计算 macro F1/Accuracy。", "",
                  "| 数据集/seed | Gate / pipeline OOS F1 | Gate / pipeline macro F1 | Gate / pipeline Accuracy | Δmacro / ΔAcc pp | Router / Expert error |",
                  "|---|---:|---:|---:|---:|---:|"])
    for r in reconciliation:
        lines.append(f"| {r['dataset']}/{r['seed']} | {100*float(r['gate_oos_f1_recomputed']):.2f} / {100*float(r['full_pipeline_oos_f1']):.2f} | "
                     f"{100*float(r['gate_macro_f1']):.2f} / {100*float(r['full_pipeline_macro_f1']):.2f} | "
                     f"{100*float(r['gate_accuracy']):.2f} / {100*float(r['full_pipeline_accuracy']):.2f} | "
                     f"{100*float(r['macro_f1_delta']):+.2f} / {100*float(r['accuracy_delta']):+.2f} | "
                     f"{100*float(r['router_error_rate']):.2f} / {100*float(r['expert_error_rate']):.2f} |")
    lines.extend(["", "Router error 是 accepted Known 中的 domain 错误比例；Expert error 是 accepted Known 中 domain 正确但 intent 错误的比例，两者使用相同分母且不重复计数。Gate Known Recall 指 Known 被接受的比例，不是 Known 分类正确率。",
                  "这些下游错误总数不等于相对最近中心分类新增的错误数：Router/Expert 也可能修复原来的 intent 错误；表中的 ΔAccuracy 才是净变化。",
                  "逐样本 CUDA 对账另保存 downstream_repaired_count/regressed_count，满足 ΔAccuracy=(修复数−新增错误数)/测试样本数。历史保存值和本次重算值分别保留；replay_minus_saved_pipeline_* 报告差异，不通过放宽断言假装精确复现。",
                  "OOS False Acceptance 分母为所有真实 OOS。legacy known_macro_f1 仅在真实 Known 子集上计算，不等于全测试样本上的 Known-class macro F1，本表统一使用包含 OOS 的 macro F1。",
                  "对账证据：" + ", ".join(sorted({r["pipeline_metrics_evidence"] for r in reconciliation})) + "。无 --replay-pipeline 的重建保留已完成的逐样本 CUDA 对账，不降级为旧聚合证据。", "",
                  "## 判因与失败实验", "",
                  "H0/H1 协议与表示均有差异，现有对照不能把论文差距定量归因给其中某一项。当前 semantic_gate_enabled=False 的 H1 实现中，Router/Expert 不改 is_oos；直接运行核验的是该配置下的 Gate 编码/边界接线一致性，不能外推到历史 semantic gate 或其他 pipeline。",
                  "用户指出整体 pipeline 指标比单独 Gate 低：该现象成立，必须与 OOS F1 单独区分。原始 Gate prediction 重算与 full pipeline 逐 seed 对账见 historical_gate_pipeline_metric_reconciliation.csv。例如 CLINC seed42 的 Gate OOS Recall=96.37%、OOS F1=89.56%、macro F1=81.12%，full pipeline OOS F1=89.56%、macro F1 历史保存值为77.21%（本次重算见上表）；下游确实降低了整体分类表现。当前报告不能据 OOS 一致性断言整个 pipeline 无损失，也不能在未定位用户指向的另一组高 OOS F1 记录时把差异归因为指标混用。历史聚合与当前重算的小幅分类差异尚未定位到具体运行时原因，不能声称 byte-identical 重现。",
                  "统一增加 K、扩展 λ/threshold、细 threshold 和 per-sphere calibration 的历史负结果仍保留；本轮只检验 train-only 稀疏拆分，不能从更多中心推断一定收益。",
                  "如果 CLINC 自适应结果仍低于论文，应根据固定 score oracle 与 intent 错误决定表示训练实验，停止无目的扩大边界网格。", "",
                  "## 证据与复现", "",
                  "- `results/analysis/historical_trainable_adaptive_centers/MANIFEST.json`：设备与协议。",
                  "- 每个 dataset/seed 的 `center_assignments.json`、`selection_lock.json`、`validation_workpoints.csv`、`test_workpoints.csv`：所有策略映射、选择锁和候选指标。",
                  "- `selected_validation_per_seed.csv`、`selected_test_per_seed.csv`、`selected_test_summary.csv`：锁定配置与完整确认。",
                  "- `strategy_validation_selected_test.csv`、`ranking_diagnostics.csv`、`intent_errors.csv`：逐策略和聚合错误诊断。",
                  "- `python tools/analysis/build_historical_trainable_adaptive_centers_report.py --replay-pipeline`：仅用现有 checkpoint 重算九个单元的逐样本对账，不训练。",
                  "- `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python scripts/experiments/run_historical_trainable_adaptive_centers.py --device cuda`。历史目录已存在时须使用新的 `--output-root`，不覆盖。", ""])
    (ROOT / "docs/analysis/historical_trainable_adaptive_centers_presentation.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"selected_rows": len(selected), "strategy_rows": len(strategies), "status": manifest["status"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-pipeline", action="store_true")
    build(parser.parse_args().replay_pipeline)
