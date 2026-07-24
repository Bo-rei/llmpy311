"""Resume protocol_v2 Gate cells whose immutable output manifests are absent."""

import sys

from s2c.experiments.runner import main


if __name__ == "__main__":
    raise SystemExit(main(["--resume", *sys.argv[1:]]))
