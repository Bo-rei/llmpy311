"""多簇/OOS 研究工作流的统一命令行入口。"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from . import analysis, baselines, export, runner


COMMANDS: dict[str, tuple[Callable[[Sequence[str] | None], None], str]] = {
    "grid": (runner.main, "运行 fixed/tuned KIR x K 主网格或单个几何 Gate 单元"),
    "baseline": (baselines.main, "运行统一 MiniLM 表征上的受控 Gate-only Baseline"),
    "analyze": (analysis.main, "运行稳定性、near/far OOS、overlap 与表征分析"),
    "export": (export.main, "导出论文表格、selected K 和完整性审计"),
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

