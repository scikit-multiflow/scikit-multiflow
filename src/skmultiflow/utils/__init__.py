"""
The :mod:`skmultiflow.utils` module contains contains utility methods for `scikit-multiflow`.
"""

from .validation import check_random_state
from .validation import check_weights
from .utils import normalize_values_in_dict
from .utils import get_dimensions
from .utils import is_scalar_nan
from .data_structures import FastBuffer
from .data_structures import FastComplexBuffer
from ._show_versions import show_versions

__all__ = ["check_random_state", "check_weights", "get_dimensions", 'is_scalar_nan',
           "normalize_values_in_dict", "FastBuffer", "FastComplexBuffer", "show_versions"]
