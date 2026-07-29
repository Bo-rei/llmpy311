"""Run the independent E3-A partition-control experiment layer."""

from protocol_v2.experiments.mechanism_runner import main


if __name__ == "__main__":
    raise SystemExit(main(["partition-control", *__import__("sys").argv[1:]]))

