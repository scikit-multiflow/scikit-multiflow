"""
The :mod:`skmultiflow.metrics` module includes performance metrics.
"""

from .measure_collection import ClassificationMeasurements
from .measure_collection import RegressionMeasurements
from .measure_collection import MultiOutputMeasurements
from .measure_collection import MultiTargetRegressionMeasurements
from .measure_collection import WindowClassificationMeasurements
from .measure_collection import WindowRegressionMeasurements
from .measure_collection import WindowMultiOutputMeasurements
from .measure_collection import WindowMultiTargetRegressionMeasurements
from .measure_collection import ConfusionMatrix
from .measure_collection import MOLConfusionMatrix
from .measure_collection import hamming_score
from .measure_collection import exact_match
from .measure_collection import j_index

__all__ = ["ClassificationMeasurements", "RegressionMeasurements",
           "MultiOutputMeasurements", "MultiTargetRegressionMeasurements",
           "WindowClassificationMeasurements", "WindowRegressionMeasurements",
           "WindowMultiOutputMeasurements",
           "WindowMultiTargetRegressionMeasurements",
           "ConfusionMatrix", "MOLConfusionMatrix", "hamming_score",
           "exact_match", "j_index"]
