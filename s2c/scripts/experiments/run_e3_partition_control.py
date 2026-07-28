"""Run the independent E3-A partition-control experiment layer."""

from s2c.experiments.mechanism_runner import main


if __name__ == "__main__":
    raise SystemExit(main(["partition-control", *__import__("sys").argv[1:]]))

