"""A-GRASPQ-FS: adaptive GRASP-based feature selection.

The public API intentionally follows scikit-learn conventions so the selector can
be used as a standalone transformer or inside ``Pipeline`` objects.
"""

from ._version import __version__
from .selector import AGraspQFeatureSelector

__all__ = ["AGraspQFeatureSelector", "__version__"]
