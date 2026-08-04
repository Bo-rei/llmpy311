"""Risk-calibrated adaptive multi-centre boundary learning (RC-AMBL).

The implementation is intentionally isolated from the frozen E2/E3 detector.
It uses the same protocol views and frozen embedding cache, but owns its
structure-selection, calibration and evidence contracts.
"""

from .contracts import AdaptiveConfig, CenterSpec, SplitOperation
from .partition import pca_median_split, bootstrap_split_stability
from .evidence import EvidenceModel
from .selection import fit_rc_ambl

__all__ = [
    "AdaptiveConfig",
    "CenterSpec",
    "SplitOperation",
    "EvidenceModel",
    "pca_median_split",
    "bootstrap_split_stability",
    "fit_rc_ambl",
]
