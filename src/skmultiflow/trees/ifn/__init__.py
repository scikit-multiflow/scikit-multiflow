"""
The :mod:`skmultiflow.trees.ifn` module includes support classes and method
for Incremental Fuzzy Network (IFN) methods.
"""

from .data_processing import DataProcessor
from .data_processing_multi import DataProcessorMulti
from .ifn_network import IfnNetwork, HiddenLayer
from .ifn_network_multi import IfnNetworkMulti
from .iolin import IncrementalOnlineNetwork
from .meta_learning import MetaLearning

__all__ = ["DataProcessor", "DataProcessorMulti", "HiddenLayer", "IfnNetwork", "HiddenLayer",
           "IfnNetworkMulti", "IncrementalOnlineNetwork", "MetaLearning"]
