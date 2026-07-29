"""
Minimal Deep SVDD Gate - Strict ICML 2018 Implementation

严格对齐 Ruff et al., "Deep One-Class Classification" (ICML 2018)
Soft-boundary Deep SVDD (Eq. 3)

禁止任何偏离论文的工程优化。
"""

from .model import MinimalSVDDGate
from .loss import SVDDLoss

__all__ = ['MinimalSVDDGate', 'SVDDLoss']
