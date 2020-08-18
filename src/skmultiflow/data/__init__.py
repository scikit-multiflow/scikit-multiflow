"""
The :mod:`skmultiflow.data` module contains data stream methods including methods for
batch-to-stream conversion and generators.
"""

from src.skmultiflow.data.generator.agrawal_generator import AGRAWALGenerator
from src.skmultiflow.data.generator.hyper_plane_generator import HyperplaneGenerator
from src.skmultiflow.data.generator.regression_generator import RegressionGenerator
from src.skmultiflow.data.generator.stagger_generator import STAGGERGenerator
from .synth import make_logical
from src.skmultiflow.data.generator.waveform_generator import WaveformGenerator
from .time_manager import TimeManager

__all__ = ["DataStream", "TemporalDataStream", "SimpleStream", "AGRAWALGenerator",
           "ConceptDriftStream", "HyperplaneGenerator", "LEDGenerator", "LEDGeneratorDrift",
           "MIXEDGenerator", "MultilabelGenerator", "RandomRBFGenerator",
           "RandomRBFGeneratorDrift", "RandomTreeGenerator", "RegressionGenerator", "SEAGenerator",
           "SineGenerator", "STAGGERGenerator", "make_logical", "WaveformGenerator", "TimeManager",
           "AnomalySineGenerator"]
