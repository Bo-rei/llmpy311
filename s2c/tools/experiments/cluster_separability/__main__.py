"""多簇/OOS 研究工作流的统一命令行入口。"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from . import (
    analysis,
    baselines,
    export,
    runner,
    v20_adaptive_boundary,
    v20_analysis,
    v20_end_to_end,
    v20_random_partition,
    v21_cluster_casebook,
    v21_representation_adaptation,
    v21_semantic_probe,
)


COMMANDS: dict[str, tuple[Callable[[Sequence[str] | None], None], str]] = {
    "grid": (runner.main, "运行 fixed/tuned KIR x K 主网格或单个几何 Gate 单元"),
    "baseline": (baselines.main, "运行统一 MiniLM 表征上的受控 Gate-only Baseline"),
    "analyze": (analysis.main, "运行稳定性、near/far OOS、overlap 与表征分析"),
    "export": (export.main, "导出论文表格、selected K 和完整性审计"),
    "v20-analysis": (v20_analysis.main, "读取 v19 产物，生成 v20 的 K 选择、near-OOS 和效率分析"),
    "v20-random": (v20_random_partition.main, "运行 KMeans 与随机分簇的受控对照"),
    "v20-end-to-end": (v20_end_to_end.main, "准备/执行 Gate 替换的端到端传递实验"),
    "v20-adaptive": (v20_adaptive_boundary.main, "运行同表征自适应局部边界 Baseline"),
    "v21-semantic-probe": (v21_semantic_probe.main, "诊断冻结 MiniLM 的邻域、几何与 near-OOS 表示错误"),
    "v21-casebook": (v21_cluster_casebook.main, "导出子簇代表句、关键词和人工语义审计模板"),
    "v21-adaptation": (v21_representation_adaptation.main, "预检/运行 Frozen、CE、SupCon MiniLM 表征对照"),
}


def _print_help() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tools.experiments.cluster_separability",
        description="MiniLM multiple-cluster / OOS separability workflow",
    )
    parser.add_argument("command", nargs="?", choices=tuple(COMMANDS))
    parser.print_help()
    print("\ncommands:")
    for name, (_, description) in COMMANDS.items():
        print(f"  {name:<9} {description}")
    print("\n使用 '<command> --help' 查看该阶段的完整参数。")


def main(argv: Sequence[str] | None = None) -> None:
    """只负责阶段分发；具体参数仍由单一职责模块解析。

    这种薄入口避免再复制一套 argparse 定义。新增或修改实验参数时，只需维护
    对应模块，统一入口不会与真实 runner 发生参数漂移。
    """

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        _print_help()
        return
    command = arguments.pop(0)
    if command not in COMMANDS:
        choices = ", ".join(COMMANDS)
        raise SystemExit(f"unknown command {command!r}; choose one of: {choices}")
    handler, _ = COMMANDS[command]
    handler(arguments)


if __name__ == "__main__":
    main()
