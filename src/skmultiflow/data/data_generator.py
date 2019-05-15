import pandas as pd
import numpy as np
from skmultiflow.data.base_generator import BaseGenerator

# TODO : doc for all methods

class DataGenerator(BaseGenerator):
    # TODO : inherit from pandas.DataFrame should make things easier
    """ DataGenerator

    A generator constructed from the entries of a static dataset (numpy array or pandas
    DataFrame).

    The batch generator is able to provide, as requested, a number of samples, in
    a way that old samples cannot be accessed in a later time. This is done
    so that a stream context can be correctly simulated.

    Parameters
    ----------
    data: np.ndarray or pd.DataFrame
        The features' columns and targets' columns or the feature columns
        only if they are passed separately.
    """

    def __init__(self, data, y=None, return_np=False):
        BaseGenerator.__init__(self)

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data=data)

        self.data = data
        self.sample_idx = 0

        self.return_np = return_np

    def next_sample(self, batch_size=1):
        sample = self.data.iloc[self.sample_idx:self.sample_idx + batch_size]
        self.sample_idx += batch_size
        if not sample.empty:
            return sample.values if self.return_np else sample
        else:
            return (None, None)

    def prepare_for_use(self):
        self.sample_idx = 0

    def get_info(self):
        return self.data.info()

    def has_more_samples(self):
        """ Checks if stream has more samples.

        Returns
        -------
        Boolean
            True if stream has more samples.

        """
        return self.n_remaining_samples() > 0

    def n_remaining_samples(self):
        """ Returns the estimated number of remaining samples.

        Returns
        -------
        int
            Remaining number of samples.

        """
        return (self.data.shape[0] - self.sample_idx)

