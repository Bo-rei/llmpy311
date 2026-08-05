"""Independent RACAL-v1 experiment implementation.

The package is intentionally isolated from the historical E2/E3/R1 and
adaptive-v1 implementations.  The first stage only contains a frozen E2 K=1
replay and a trainable MiniLM K=1 control; centre activation is reserved for a
later, separately registered stage.
"""

STAGE = "racal_v1"

__all__ = ["STAGE"]
