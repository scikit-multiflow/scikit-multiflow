"""
The :mod:`skmultiflow.trees.ifn` module includes support classes and method
for Incremental Fuzzy Network (IFN) methods.
"""

from .data_processing import DataProcessor
from .ifn_network import IfnNetwork, HiddenLayer
from .iolin import IncrementalOnlineNetwork
from .meta_learning import MetaLearning

__all__ = ["DataProcessor", "HiddenLayer", "IfnNetwork", "IncrementalOnlineNetwork", "MetaLearning"]
