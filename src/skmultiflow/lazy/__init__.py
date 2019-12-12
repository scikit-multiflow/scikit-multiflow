"""
The :mod:`skmultiflow.lazy` module includes lazy learning methods in which generalization of the training
data is delayed until a query is received, this is, on-demand.
"""

from .knn_classifier import KNNClassifier
from .knn_adwin import KNNAdwin
from .sam_knn import SAMKNN
from .kdtree import KDTree
from .knn_classifier import KNN   # remove in v0.7.0


__all__ = ["KNNClassifier", "KNNAdwin", "SAMKNN", "KDTree", "KNN"]
