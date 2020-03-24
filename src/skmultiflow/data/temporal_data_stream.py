import pandas as pd
import numpy as np

import warnings

from skmultiflow.data.data_stream import DataStream
from skmultiflow.utils import add_delay_to_timestamps

class TemporalDataStream(DataStream):
    """ Creates a temporal stream from a data source.

    TemporalDataStream takes the whole data set containing the `X` (features), `time` (timestamps) and `Y` (targets).

    Parameters
    ----------
    data: np.ndarray or pd.DataFrame
        The features' columns and targets' columns or the feature columns
        only if they are passed separately.
    time: np.ndarray(dtype=datetime64) or pd.Series (Default=None)
        The timestamp column of each instance. If its a pd.Series, it will
        be converted into a np.ndarray.
    sample_weight: np.ndarray or pd.Series, optional (Default=None)
        Sample weights.
    sample_delay: np.ndarray(np.datetime64), pd.DataFrame, np.timedelta64 or int, optional (Default=np.timedelta64(0,"D"))
        Samples delay in np.timedelta64 (the dateoffset difference between the event time
        and when the label is available), np.ndarray(np.datetime64) with the timestamp that the sample
        will be available or int with the delay in number of samples.
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

    ordered: bool, optional (default=True)
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
    def __init__(self, data, time=None, y=None, sample_weight=None, sample_delay=np.timedelta64(0,"D"), target_idx=-1,
                 n_targets=1, cat_features=None, name=None, ordered=True):
        # check if time is pandas dataframe or a numpy.ndarray
        if isinstance(time, pd.Series):
            self.time = pd.to_datetime(time).values
        elif isinstance(time, np.ndarray):
            self.time = np.array(time,dtype="datetime64")
        elif time is None:
            self.time = None
        else:
            raise ValueError("np.ndarray, pd.Series or None time object expected, and {} was passed".format(type(time)))
        # check if its a single delay or a delay for instance and save delay
        if isinstance(sample_delay, np.timedelta64):
            # create delays list
            self.sample_delay = add_delay_to_timestamps(time,sample_delay)
        elif isinstance(sample_delay, pd.Series):
            self.sample_delay = pd.to_datetime(sample_delay.values).values
        elif isinstance(sample_delay, np.ndarray):
            self.sample_delay = np.array(sample_delay, dtype="datetime64")
        elif isinstance(sample_delay, int):
            if self.time is not None:
                warnings.warn("'time' is not going to be used because 'sample_delay' is int. Delay by number of samples"
                              "is applied. If you want to use time delay, use np.timedelta64 for 'sample_delay'.")
            self.time = np.arange(0, self.time.shape[0])
            self.sample_delay = np.arange(0 + sample_delay, self.time.shape[0] + sample_delay)
        else:
            raise ValueError("np.ndarray(np.datetime64), pd.Series, np.timedelta64 or int sample_delay object expected, and {} was passed".format(type(sample_delay)))

        # save sample weights if available
        if sample_weight is not None:
            self.sample_weight = sample_weight
        else:
            self.sample_weight = None
        # if data is not ordered, order it
        if not ordered:
            # order data based on self.time
            data = data[np.argsort(self.time)]
            # order y based on self.time
            y = y[np.argsort(self.time)]
            # order sample_weight if available
            if self.sample_weight is not None:
                self.sample_weight = self.sample_weight[np.argsort(self.time)]
            # order sample_delay, check if not single delay
            self.sample_delay = self.sample_delay[np.argsort(self.time)]
            # order self.time
            self.time.sort()
        super().__init__(data, y, target_idx, n_targets, cat_features, name)
    
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
            Returns the next batch_size instances (sample_x, sample_time, sample_y, sample_weight (if available), sample_delay (if available)).
            For general purposes the return can be treated as a numpy.ndarray.
        """
        self.sample_idx += batch_size
        
        try:

            self.current_sample_x = self.X[self.sample_idx - batch_size:self.sample_idx, :]
            self.current_sample_y = self.y[self.sample_idx - batch_size:self.sample_idx, :]
            self.current_sample_time = self.time[self.sample_idx - batch_size:self.sample_idx]
            self.current_sample_delay = self.sample_delay[self.sample_idx - batch_size:self.sample_idx]

            if self.n_targets < 2:
                self.current_sample_y = self.current_sample_y.flatten()

            # create base output
            output = [self.current_sample_x, self.current_sample_time, self.current_sample_delay, self.current_sample_y]

            # check if sampe_weight is available
            if self.sample_weight is not None:
                self.current_sample_weight = self.sample_weight[self.sample_idx - batch_size:self.sample_idx, :]
                # add to output
                output.append(self.current_sample_weight)

        except IndexError:
            self.current_sample_x = None
            self.current_sample_y = None
            self.current_sample_time = None
            self.current_sample_delay = None

            # create base output
            output = [self.current_sample_x, self.current_sample_time, self.current_sample_delay, self.current_sample_y]
            # check if sampe_weight is available
            if self.sample_weight is not None:
                self.current_sample_weight = None
                output.append(self.current_sample_weight)
            
        return output