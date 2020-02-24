import pandas as pd
import numpy as np

import warnings

from skmultiflow.data.data_stream import DataStream

# implement temporal data stream
class TemporalDataStream(DataStream):
    """ Creates a temporal stream from a data source.

    TemporalDataStream takes the whole data set containing the `X` (features), `time` (timestamps) and `Y` (targets).

    Parameters
    ----------
    data: np.ndarray or pd.DataFrame (Default=None)
        The features' columns and targets' columns or the feature columns
        only if they are passed separately.
    time: np.ndarray(dtype=datetime64) or pd.DataFrame (Default=None)
        The timestamp column of each instance. If its a np.ndarray, it will
        be converted into a pandas datetime dataframe. 
    y: np.ndarray or pd.DataFrame, optional (Default=None)
        The targets' columns.

    target_idx: int, optional (default=-1)
        The column index from which the targets start.

    n_targets: int, optional (default=1)
        The number of targets.

    cat_features: list, optional (default=None)
        A list of indices corresponding to the location of categorical features.

    name: str, optional (default=None)
        A string to id the data.

    ordered: boolean, optional (default=True)
        If True, consider that data, time and y are already ordered by timestamp.
        Otherwise, the data is ordered based on `time` timestamps.

    allow_nan: bool, optional (default=False)
        If True, allows NaN values in the data. Otherwise, an error is raised.

    Notes
    -----
    The stream object provides upon request a number of samples, in a way such that old samples cannot be accessed
    at a later time. This is done to correctly simulate the stream context.

    """
    # includes time as datetime
    def __init__(self, data, time, y=None, target_idx=-1, n_targets=1, cat_features=None, name=None, ordered=True):
        # check if time is pandas dataframe or a numpy.ndarray
        if isinstance(time, pd.Series):
            self.time = pd.to_datetime(time)
        elif isinstance(time, np.ndarray):
            self.time = pd.to_datetime(time)
        else:
            raise ValueError("np.ndarray or pd.Series time object expected, and {} was passed".format(type(time)))
        # if data is not ordered, order it
        if not ordered:
            # order data based on self.time
            data = data[np.argsort(self.time)]
            # order y based on self.time
            y = y[np.argsort(self.time)]
            # order self.time
            self.time = self.time.sort_values()
        super().__init__(data, y, target_idx, n_targets, cat_features, name)
    
    # get next sample, returning sample_x, sample_time and sample_y
    def next_sample(self, batch_size=1):
        """ next_sample
        If there is enough instances to supply at least batch_size samples, those
        are returned. If there aren't a tuple of (None, None) is returned.
        Parameters
        ----------
        batch_size: int
            The number of instances to return.
        Returns
        -------
        tuple or tuple list
            Returns the next batch_size instances.
            For general purposes the return can be treated as a numpy.ndarray.
        """
        self.sample_idx += batch_size
        
        try:

            self.current_sample_x = self.X[self.sample_idx - batch_size:self.sample_idx, :]
            self.current_sample_y = self.y[self.sample_idx - batch_size:self.sample_idx, :]
            self.current_sample_time = self.time[self.sample_idx - batch_size:self.sample_idx]
            if self.n_targets < 2:
                self.current_sample_y = self.current_sample_y.flatten()

        except IndexError:
            self.current_sample_x = None
            self.current_sample_y = None
            self.current_sample_time = None
            
        return self.current_sample_x, self.current_sample_time, self.current_sample_y
    
    def get_temporal_information(self):
        return self.time