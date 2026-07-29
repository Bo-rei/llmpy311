"""Summarize the isolated MiniLM training pilot."""

from protocol_v2.experiments.minilm_training import main


if __name__ == "__main__":
    raise SystemExit(main(["summarize", *__import__("sys").argv[1:]]))
