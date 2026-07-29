"""Run the train/calibration-only E3-B/C diagnostics."""

from protocol_v2.experiments.mechanism_runner import main


if __name__ == "__main__":
    raise SystemExit(main(["cluster-diagnostics", *__import__("sys").argv[1:]]))

