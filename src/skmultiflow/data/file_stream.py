import os
import pandas as pd
import numpy as np
import warnings
from skmultiflow.data.data_generator import DataGenerator


class FileStream(DataGenerator):
    """ FileStream

    A stream generated from the entries of a file. For the moment only
    csv files are supported, but the idea is to support different formats,
    as long as there is a function that correctly reads, interprets, and
    returns a pandas' DataFrame or numpy.ndarray with the data.

    The stream is able to provide, as requested, a number of samples, in
    a way that old samples cannot be accessed in a later time. This is done
    so that a stream context can be correctly simulated.

    Parameters
    ----------
    filepath:
        Path to the data file

    target_idx: int, optional (default=-1)
        The column index from which the targets start.

    n_targets: int, optional (default=1)
        The number of targets.

    cat_features_idx: list, optional (default=None)
        A list of indices corresponding to the location of categorical features.
    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.file_stream import FileStream
    >>> # Setup the stream
    >>> stream = FileStream('skmultiflow/data/datasets/sea_stream.csv')
    >>> stream.prepare_for_use()
    >>> # Retrieving one sample
    >>> stream.next_sample()
    (array([[0.080429, 8.397187, 7.074928]]), array([0]))
    >>> # Retrieving 10 samples
    >>> stream.next_sample(10)
    (array([[1.42074 , 7.504724, 6.764101],
        [0.960543, 5.168416, 8.298959],
        [3.367279, 6.797711, 4.857875],
        [9.265933, 8.548432, 2.460325],
        [7.295862, 2.373183, 3.427656],
        [9.289001, 3.280215, 3.154171],
        [0.279599, 7.340643, 3.729721],
        [4.387696, 1.97443 , 6.447183],
        [2.933823, 7.150514, 2.566901],
        [4.303049, 1.471813, 9.078151]]),
        array([0, 0, 1, 1, 1, 1, 0, 0, 1, 0]))
    >>> stream.n_remaining_samples()
    39989
    >>> stream.has_more_samples()
    True

    """
    _CLASSIFICATION = 'classification'
    _REGRESSION = 'regression'

    warnings.warn("FileStream will be deprecated in the future, use skmultiflow.data.DataGenerator instead")

    def __init__(self, filepath, target_idx=-1, n_targets=1, cat_features_idx=None, return_np=True, **kwargs):
        data = pd.read_csv(filepath)
        super().__init__(data, return_np=False, **kwargs)

        self.filepath = filepath
        self.n_targets = n_targets
        self.target_idx = target_idx
        self.cat_features_idx = [] if cat_features_idx is None else cat_features_idx

        self.feature_names = list(self.data.columns)[:target_idx]
        self.target_names = list(self.data.columns)[target_idx:]
        self.n_features = len(self.feature_names)
        self.n_cat_features = len(self.cat_features_idx)
        self.n_num_features = self.n_features - self.n_cat_features
        self._return_np = return_np

        # Automatically infer target_idx if not passed in multi-output problems
        if self.n_targets > 1 and self.target_idx == -1:
            self.target_idx = -self.n_targets

        self.X = self.data[self.feature_names]
        self.y = self.data[self.target_names]
        if len(self.y.shape) == 2:
            self.y = self.y.iloc[:, 0]
        self.target_values = list(self.y.unique())
        self.task_type = None
        self.n_classes = len(self.target_values)
        self.name = os.path.split(self.filepath)[-1]

    def next_sample(self, batch_size=1):
        return self._unzip_data(super().next_sample(batch_size=batch_size))

    def last_sample(self):
        return self._unzip_data(super().last_sample())

    def _unzip_data(self, sample):
        if sample is not None:
            X, y = sample[self.feature_names], sample[self.target_names]
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
            return (X.values, y.values) if self._return_np else (X, y)
        else:
            return None
