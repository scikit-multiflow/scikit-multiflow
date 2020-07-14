"""
The :mod:`skmultiflow.trees.nodes` module includes learning and split node
implementations for the hoeffding trees.
"""

# Base class nodes
from .base import FoundNode
from .base import Node
from .base import SplitNode
from .base import LearningNode
from .base import ActiveLeaf
from .base import InactiveLeaf
# Hoeffding Tree classifier nodes
from ._htc_nodes import ActiveLeafClass
from ._htc_nodes import LearningNodeMC
from ._htc_nodes import LearningNodeNB
from ._htc_nodes import LearningNodeNBA
from ._htc_nodes import ActiveLearningNodeMC
from ._htc_nodes import InactiveLearningNodeMC
from ._htc_nodes import ActiveLearningNodeNB
from ._htc_nodes import ActiveLearningNodeNBA
# Hoeffding Adaptive Tree classifier nodes
from ._hatc_nodes import AdaNode
from ._hatc_nodes import AdaSplitNode                      # TODO: verify name
from ._hatc_nodes import AdaLearningNode                   # TODO: verify name
# Adaptive Random Forest classifier nodes
from ._arf_htc_nodes import RandomActiveLeafClass
from ._arf_htc_nodes import RandomActiveLearningNodeMC
from ._arf_htc_nodes import RandomActiveLearningNodeNB
from ._arf_htc_nodes import RandomActiveLearningNodeNBA
# Extremely Fast Decision Tree classifier nodes
from ._efdtc_nodes import EFDTSplitNode
from ._efdtc_nodes import EFDTActiveLearningNodeMC
from ._efdtc_nodes import EFDTInactiveLearningNodeMC
from ._efdtc_nodes import EFDTActiveLearningNodeNB
from ._efdtc_nodes import EFDTActiveLearningNodeNBA
# Label combination classifier nodes
from ._lc_htc_nodes import LCActiveLearningNodeMC
from ._lc_htc_nodes import LCInactiveLearningNodeMC
from ._lc_htc_nodes import LCActiveLearningNodeNB
from ._lc_htc_nodes import LCActiveLearningNodeNBA
# Hoeffding Tree regressor nodes
from ._htr_nodes import ActiveLeafRegressor
from ._htr_nodes import LearningNodeMean
from ._htr_nodes import LearningNodePerceptron
from ._htr_nodes import ActiveLearningNodeMean
from ._htr_nodes import ActiveLearningNodePerceptron
from ._htr_nodes import InactiveLearningNodeMean
from ._htr_nodes import InactiveLearningNodePerceptron
# Hoeffding Adaptive Tree regressor nodes
from ._hatr_nodes import AdaSplitNodeRegressor
from ._hatr_nodes import AdaActiveLearningNodeRegressor
# Adaptive Random Forest regressor nodes
from ._arf_htr_nodes import RandomActiveLeafRegressor
from ._arf_htr_nodes import RandomActiveLearningNodeMean
from ._arf_htr_nodes import RandomActiveLearningNodePerceptron
from ._isouptr_nodes import LearningNodePerceptronMultiTarget
from ._isouptr_nodes import LearningNodeAdaptiveMultiTarget
from ._isouptr_nodes import ActiveLearningNodePerceptronMultiTarget
from ._isouptr_nodes import ActiveLearningNodeAdaptiveMultiTarget
from ._isouptr_nodes import InactiveLearningNodePerceptronMultiTarget
from ._isouptr_nodes import InactiveLearningNodeAdaptiveMultiTarget
from ._sst_htr_nodes import SSTLearningNode
from ._sst_htr_nodes import SSTLearningNodeAdaptive
from ._sst_htr_nodes import SSTActiveLearningNode
from ._sst_htr_nodes import SSTInactiveLearningNode
from ._sst_htr_nodes import SSTActiveLearningNodeAdaptive
from ._sst_htr_nodes import SSTInactiveLearningNodeAdaptive


__all__ = ["FoundNode", "Node", "SplitNode", "LearningNode", "ActiveLeaf", "InactiveLeaf",
           "AdaNode", "ActiveLeafClass", "LearningNodeMC", "LearningNodeNB", "LearningNodeNBA",
           "ActiveLearningNodeMC", "InactiveLearningNodeMC", "ActiveLearningNodeNB",
           "ActiveLearningNodeNBA", "RandomActiveLeafClass", "RandomActiveLearningNodeMC",
           "RandomActiveLearningNodeNB", "RandomActiveLearningNodeNBA", "AdaSplitNode",
           "AdaLearningNode", "EFDTActiveLeaf", "EFDTSplitNode", "EFDTActiveLearningNodeMC",
           "EFDTInactiveLearningNodeMC", "EFDTActiveLearningNodeNB", "EFDTActiveLearningNodeNBA",
           "LCActiveLearningNodeMC", "LCInactiveLearningNodeMC", "LCActiveLearningNodeNB",
           "LCActiveLearningNodeNBA", "ActiveLeafRegressor", "LearningNodeMean",
           "LearningNodePerceptron", "ActiveLearningNodeMean", "ActiveLearningNodePerceptron",
           "InactiveLearningNodeMean", "InactiveLearningNodePerceptron",
           "RandomActiveLeafRegressor", "RandomActiveLearningNodeMean",
           "RandomActiveLearningNodePerceptron", "AdaSplitNodeRegressor",
           "AdaActiveLearningNodeRegressor", "LearningNodePerceptronMultiTarget",
           "LearningNodeAdaptiveMultiTarget",  "ActiveLearningNodePerceptronMultiTarget",
           "ActiveLearningNodeAdaptiveMultiTarget", "InactiveLearningNodePerceptronMultiTarget",
           "InactiveLearningNodeAdaptiveMultiTarget", "SSTLearningNode", "SSTLearningNodeAdaptive",
           "SSTActiveLearningNode", "SSTActiveLearningNodeAdaptive", "SSTInactiveLearningNode",
           "SSTInactiveLearningNodeAdaptive"]
