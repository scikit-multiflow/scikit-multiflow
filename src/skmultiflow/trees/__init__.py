"""
The :mod:`skmultiflow.trees` module includes learning methods based on trees.
"""
from . import attribute_observer
from . import attribute_test
from . import nodes
from . import split_criterion
from .hoeffding_tree import HoeffdingTreeClassifier
from .hoeffding_adaptive_tree import HoeffdingAdaptiveTreeClassifier
from .extremely_fast_decision_tree import ExtremelyFastDecisionTreeClassifier
from .label_combination_hoeffding_tree import LabelCombinationHoeffdingTree
from .hoeffding_tree_regressor import HoeffdingTreeRegressor
from .hoeffding_adaptive_tree_regressor import HoeffdingAdaptiveTreeRegressor
from .multi_target_regression_hoeffding_tree import MultiTargetRegressionHoeffdingTree
from .stacked_single_target_hoeffding_tree_regressor import StackedSingleTargetHoeffdingTreeRegressor
from .hoeffding_tree import HoeffdingTree   # remove in v0.7.0
from .hoeffding_adaptive_tree import HAT   # remove in v0.7.0
from .extremely_fast_decision_tree import HATT   # remove in v0.7.0"
from .label_combination_hoeffding_tree import LCHT   # remove in v0.7.0"
from .hoeffding_tree_regressor import RegressionHoeffdingTree   # remove in v0.7.0"
from .hoeffding_adaptive_tree_regressor import RegressionHAT   # remove in v0.7.0"

__all__ = ["attribute_observer", "attribute_test", "nodes", "split_criterion",
           "HoeffdingTreeClassifier", "HoeffdingAdaptiveTreeClassifier", "ExtremelyFastDecisionTreeClassifier",
           "LabelCombinationHoeffdingTree", "HoeffdingTreeRegressor", "HoeffdingAdaptiveTreeRegressor",
           "HoeffdingTree", "HAT", "HATT", "LCHT", "RegressionHoeffdingTree",
           "RegressionHAT", "MultiTargetRegressionHoeffdingTree",
           "StackedSingleTargetHoeffdingTreeRegressor"]

