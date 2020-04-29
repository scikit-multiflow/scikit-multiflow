import numpy as np
import pandas as pd


class TimeManager(object):
    """ TimeManager

    Manage instances that are related to a timestamp and
    a delay.

    Parameters
    ----------
    timestamp: np.datetime64
        Current timestamp of the stream. This timestamp is always
        updated as the stream is processed to simulate when
        the labels will be available for each sample.

    """

    _columns = ["X", "y_real", "y_pred", "weight", "arrival_time", "available_time"]
    _sample_weight = True

    def __init__(self, timestamp):
        # get initial timestamp
        self.timestamp = timestamp
        # create dataframe to save time data
        # X = features
        # y_real = real label
        # y_pred = predicted label for each model being evaluated
        # arrival_time = arrival timestamp of the sample
        # available_time = timestamp when the sample label will be available
        self.queue = pd.DataFrame(columns=self._columns)
        # transform queue into datetime if timestamps are not int
        if isinstance(self.timestamp, int):
            self.queue['arrival_time'] = pd.to_datetime(self.queue['arrival_time'])
            self.queue['available_time'] = pd.to_datetime(self.queue['available_time'])

    def _sort_queue(self):
        # sort values by available_time
        self.queue = self.queue.sort_values(by='available_time', ignore_index=True)

    def _cleanup_samples(self, batch_size):
        # check if has batch_size samples to be removed
        if self.queue.shape[0] - batch_size >= 0:
            # drop samples that were already used
            self.queue = self.queue.iloc[batch_size:]
            # reset indexes
            self.queue = self.queue.reset_index(drop=True)
        else:
            # drop all samples
            self.queue = self.queue.iloc[0:0]

    def update_timestamp(self, timestamp):
        """ update_timestamp

        Update current timestamp of the stream.

        Parameters
        ----------
        timestamp: datetime64
            Current timestamp of the stream. This timestamp is always
            updated as the stream is processed to simulate when
            the labels will be available for each sample.

        """

        self.timestamp = timestamp

    def get_available_samples(self):
        """ get_available_samples

        Get available samples of the stream, i.e., samples that have
        their labels available (available_time <= timestamp).

        Returns
        -------
        tuple
            A tuple containing the data, their real labels and predictions.

        """

        # get samples that have label available
        samples = self.queue[self.queue['available_time'] <= self.timestamp]
        # remove these samples from queue
        self.queue = self.queue[self.queue['available_time'] > self.timestamp]
        # return X, y_real and y_pred for the unqueued samples
        return samples["X"], samples["y_real"], samples["y_pred"]

    def update_queue(self, X, y_real, y_pred, weight, arrival_time, available_time):
        # check if weight is None to create a list
        if weight is None:
            # set _sample_weight as False, indicating that there is no weight
            self._sample_weight = False
            weight = np.full(X.shape[0], None)
        # create daraframe for current samples
        frame = pd.DataFrame(list(zip(X, y_real, y_pred, weight, arrival_time, available_time)),
                             columns=self._columns)
        # append new data to queue
        self.queue = self.queue.append(frame)
        # sort queue
        self._sort_queue()

    def has_more_samples(self):
        """ Checks if queue has more samples.

        Returns
        -------
        Boolean
            True if queue has more samples.

        """

        return self.queue.shape[0] > 0

    def next_sample(self, batch_size=1):
        """ Returns next sample from the queue.

        If there is enough instances to supply at least batch_size samples, those
        are returned. Otherwise, the remaining are returned.

        Parameters
        ----------
        batch_size: int (optional, default=1)
            The number of instances to return.

        Returns
        -------
        tuple or tuple list
            Returns the next batch_size instances.
            For general purposes the return can be treated as a numpy.ndarray.

        """

        if self.queue.shape[0] - batch_size >= 0:
            samples = self.queue[:batch_size]
        else:
            samples = self.queue
        # get X
        X = samples["X"]
        # get y_real
        y_real = samples["y_real"]
        # get y_pred
        y_pred = samples["y_pred"]
        # check if sample_weight are being used
        if self._sample_weight:
            weight = samples["weight"]
        else:
            weight = None
        # remove samples from queue
        self._cleanup_samples(batch_size)
        # return X, y_real, y_pred, and weight for the unqueued samples
        return X, y_real, y_pred, weight
