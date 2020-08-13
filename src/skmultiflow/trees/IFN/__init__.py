"""
The :mod:`skmultiflow.trees.IFN` module includes support classes and method
for Incremental Fuzzy Network (IFN) methods.
"""

from .data_processing import DataProcessor
from .IFN_network import IfnNetwork, IfnHiddenLayer
from .IFN_iolin import IfnIncrementalOnlineNetwork
from .IFN_meta_learning import IfnMetaLearning

__all__ = ["DataProcessor", "IfnHiddenLayer", "IfnNetwork", "IfnIncrementalOnlineNetwork", "IfnMetaLearning"]
