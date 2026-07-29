"""Compatibility import for the active protocol_v2 multi-sphere detector.

The detector implementation belongs to ``protocol_v2.gate`` because it is the
active Gate contract. Historical v19 callers keep importing this module path.
"""

from protocol_v2.gate.multi_sphere_oos_detector import (  # noqa: F401
    DetectorMetrics,
    MultiSphereOOSDetector,
    SphereConfig,
)

__all__ = ["DetectorMetrics", "MultiSphereOOSDetector", "SphereConfig"]
