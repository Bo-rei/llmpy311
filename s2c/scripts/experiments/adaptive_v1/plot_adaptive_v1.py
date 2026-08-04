#!/usr/bin/env python3
"""Generate lightweight RC-AMBL pilot plots from aggregate diagnostics."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from protocol_v2.runtime.paths import ProtocolV2Paths

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, default=None)
parser.add_argument("--output", type=Path, default=None)
args = parser.parse_args()
paths = ProtocolV2Paths.discover()
input_path = args.input or (paths.results_root / "diagnostics" / "adaptive_v1" / "per_seed_results.csv")
output = args.output or (paths.results_root / "diagnostics" / "adaptive_v1")
if args.output is None:
    output = paths.run_root / "adaptive_v1" / "contract_repair5" / "diagnostics" / "plots"
output.mkdir(parents=True, exist_ok=True)
frame = pd.read_csv(input_path)

fig, ax = plt.subplots(figsize=(8, 4.5))
frame.boxplot(column="oos_f1", by="method", ax=ax, rot=35)
ax.set_title("RC-AMBL StackOverflow OOS F1")
ax.set_xlabel("")
ax.set_ylabel("OOS F1")
fig.suptitle("")
fig.tight_layout()
fig.savefig(output / "oos_f1_comparison.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
frame.plot.scatter(x="known_recall", y="oos_f1", c="seed", colormap="viridis", ax=ax)
ax.set_title("Known Recall–OOS F1")
fig.tight_layout()
fig.savefig(output / "known_recall_vs_oos_f1.png", dpi=160)
plt.close(fig)

comparison = frame[frame["method"].isin(["RC-AMBL-KnownOnly", "RC-AMBL-ProxyOOS"])].copy()
fig, ax = plt.subplots(figsize=(7, 4))
comparison.boxplot(column="false_accept_rate", by="method", ax=ax, rot=20)
ax.set_title("False acceptance comparison")
ax.set_xlabel("")
ax.set_ylabel("False acceptance rate")
fig.suptitle("")
fig.tight_layout()
fig.savefig(output / "false_acceptance_comparison.png", dpi=160)
plt.close(fig)

diagnostic_root = output.parent if output.name == "plots" else output
ky_path = diagnostic_root / "ky_distribution.csv"
if ky_path.is_file():
    ky = pd.read_csv(ky_path)
    if not ky.empty:
        counts = ky.groupby(["method", "k_y"], as_index=False).size()
        pivot = counts.pivot(index="method", columns="k_y", values="size").fillna(0)
        pivot.plot.bar(stacked=True, figsize=(7, 4), ax=plt.gca())
        plt.title("Final K_y distribution")
        plt.xlabel("")
        plt.ylabel("Intent counts")
        plt.tight_layout()
        plt.savefig(output / "ky_distribution.png", dpi=160)
        plt.close()

ops_path = diagnostic_root / "center_operations.csv"
if ops_path.is_file():
    operations = pd.read_csv(ops_path)
    if not operations.empty:
        counts = operations.groupby(["method", "split_accepted"], as_index=False).size()
        pivot = counts.pivot(index="method", columns="split_accepted", values="size").fillna(0)
        pivot.plot.bar(stacked=True, figsize=(7, 4), ax=plt.gca())
        plt.title("RC-AMBL split operations")
        plt.xlabel("")
        plt.ylabel("Operation count")
        plt.tight_layout()
        plt.savefig(output / "split_operation_summary.png", dpi=160)
        plt.close()
