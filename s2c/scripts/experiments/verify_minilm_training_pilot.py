"""Verify provenance and completion of the isolated MiniLM pilot."""

from protocol_v2.experiments.minilm_training import main


if __name__ == "__main__":
    raise SystemExit(main(["verify", *__import__("sys").argv[1:]]))
