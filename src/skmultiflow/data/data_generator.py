import pandas as pd
import numpy as np
from skmultiflow.data.base_generator import BaseGenerator

# TODO : doc for all methods

class DataGenerator(BaseGenerator):
    """ DataGenerator

    A generator constructed from the entries of a static dataset

    The batch generator is able to provide, as requested, a number of samples, in
    a way that old samples cannot be accessed in a later time. This is done
    so that a stream context can be correctly simulated.

    Parameters
    ----------
    data:  ndarray (structured or homogeneous), Iterable, dict, or DataFrame
        the data to be passed as input. If not already a pd.DataFrame,
        a dataframe will be constucted from it
    return_np: bool
        Either `next_sample` and `last_sample` will return pandas.DataFrame or np.array
    """

    def __init__(self, data, return_np=False):
        BaseGenerator.__init__(self)

        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data=data)

        self.data = data
        self.sample_idx = 0
        self.return_np = return_np
        self._last_sample = pd.DataFrame(columns=self.data.columns)

    def next_sample(self, batch_size=1):
        """
        Pull a new batch of data out from the generator, of size batch_size

        Parameters
        ----------
        batch_size:  int
            number of elements (i.e number of rows) to pull from the generator

        Returns
        -------
        pandas.DataFrame or None
            if a sample if available, return it as a pandas DataFrame
            otherwise return None
        """
        sample = self.data.iloc[self.sample_idx:self.sample_idx + batch_size]
        self.sample_idx += batch_size
        self._last_sample = sample
        if not sample.empty:
            return sample.values if self.return_np else sample
        else:
            return None

    def last_sample(self):
        """
        Get the last batch of data returned by `next_sample`
        Returns
        -------
        pandas.DataFrame or None
            if a sample if available, return it as a pandas DataFrame
            otherwise return None
        """
        return self._last_sample

    def prepare_for_use(self):
        self.sample_idx = 0

    def get_info(self):
        return self.data.info()

    def has_more_samples(self):
        """ Checks if the generator has more samples.

        Returns
        -------
        Boolean
            True if the generator has more samples.

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

