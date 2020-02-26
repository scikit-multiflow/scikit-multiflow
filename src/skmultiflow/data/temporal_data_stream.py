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
    sample_weight: np.ndarray or pd.DataFrame, optional (Default=None)
        Sample weights.
    sample_delay: np.ndarray(pd.tseries.offsets.DateOffset) or pd.tseries.offsets.DateOffset, optional (Default=pd.tseries.offsets.DateOffset(day=0))
        Samples delay in pd.tseries.offsets.DateOffset (the dateoffset difference between the event time 
        and when the label is available).
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
    def __init__(self, data, time, y=None, sample_weight=None, sample_delay=pd.tseries.offsets.DateOffset(day=0), target_idx=-1, n_targets=1, cat_features=None, name=None, ordered=True):
        # check if time is pandas dataframe or a numpy.ndarray
        if isinstance(time, pd.Series) or isinstance(time, np.ndarray):
            self.time = pd.to_datetime(time)
        else:
            raise ValueError("np.ndarray or pd.Series time object expected, and {} was passed".format(type(time)))
        # save sample delay
        self.sample_delay = sample_delay
        # check if its a single delay or a delay for instance
        if isinstance(self.sample_delay, pd.Series) or isinstance(self.sample_delay, np.ndarray):
            self.single_delay = False
        else:
            self.single_delay = True
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
                self.sample_weight[np.argsort(self.time)]
            # order sample_delay, check if not single delay
            if not self.single_delay:
                self.sample_delay[np.argsort(self.time)]
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
            Returns the next batch_size instances (sample_x, sample_time, sample_y, sample_weight (if available), sample_delay (if available)).
            For general purposes the return can be treated as a numpy.ndarray.
        """
        self.sample_idx += batch_size
        
        try:

            self.current_sample_x = self.X[self.sample_idx - batch_size:self.sample_idx, :]
            self.current_sample_y = self.y[self.sample_idx - batch_size:self.sample_idx, :]
            self.current_sample_time = self.time[self.sample_idx - batch_size:self.sample_idx]

            # check if its a single delay
            if self.single_delay:
                # create list with same delay for each instance
                self.current_sample_delay = np.full(batch_size, self.sample_delay)
            else:
                # get delays for each instance
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
    
    def get_temporal_information(self):
        return self.time