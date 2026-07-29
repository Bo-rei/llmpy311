"""Audit the frozen StackOverflow K=1/K=2 Gate path."""

from protocol_v2.experiments.minilm_training import main


if __name__ == "__main__":
    raise SystemExit(main(["audit", *__import__("sys").argv[1:]]))
