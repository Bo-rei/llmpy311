"""Resume protocol_v2 Gate cells whose immutable output manifests are absent."""

import sys

from protocol_v2.experiments.runner import main


if __name__ == "__main__":
    raise SystemExit(main(["--resume", *sys.argv[1:]]))
