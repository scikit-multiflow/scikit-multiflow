"""
The :mod:`skmultiflow.trees` module includes learning methods based on trees.
"""

from .attribute_split_suggestion import AttributeSplitSuggestion
from .hoeffding_tree import HoeffdingTree
from .hoeffding_adaptive_tree import HAT
from .hoeffding_anytime_tree import HATT
from .arf_hoeffding_tree import ARFHoeffdingTree
from .lc_hoeffding_tree import LCHT
from .regression_hoeffding_tree import RegressionHoeffdingTree
from .regression_hoeffding_adaptive_tree import RegressionHAT
from .multi_target_regression_hoeffding_tree import MultiTargetRegressionHoeffdingTree
from .stacked_single_target_hoeffding_tree_regressor import StackedSingleTargetHoeffdingTreeRegressor

__all__ = ["AttributeSplitSuggestion", "HoeffdingTree", "HAT", "HATT", "ARFHoeffdingTree", "LCHT",
           "RegressionHoeffdingTree", "RegressionHAT", "MultiTargetRegressionHoeffdingTree",
           "StackedSingleTargetHoeffdingTreeRegressor"]
