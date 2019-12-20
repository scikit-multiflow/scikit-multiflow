import numpy as np

from skmultiflow.transform.base_transform import StreamTransform
from skmultiflow.utils import FastBuffer, get_dimensions


class MinmaxScaler(StreamTransform):
    """ Transform features by scaling each feature to a given range.
    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between zero and one.
    For the training set we consider a window of a given length.

    Parameters
    ----------
    window_size: int (Default: 200)
        Defines the window size to compute min and max values.

    Examples
    --------
    """

    def __init__(self, window_size=200):
        super().__init__()
        self.window_size = window_size
        self.window = None

        self.__configure()

    def __configure(self):
        self.window = FastBuffer(max_size=self.window_size)

    def transform(self, X):
        """ transform

        Does the transformation process in the samples in X.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The sample or set of samples that should be transformed.

        """
        r, c = get_dimensions(X)
        for i in range(r):
            row = np.copy([X[i][:]])
            for j in range(c):
                value = X[i][j]
                min = self._get_min(j)
                max = self._get_max(j)
                if((max-min)==0):
                    transformed=0
                else:
                    X_std = (value - min) / (max - min)
                    transformed = X_std * (max - min) + min
                X[i][j] = transformed
            self.window.add_element(row)
        return X

    def _get_min(self, column_index):
        min = 0.
        if not self.window.is_empty():
            min = np.nanmin(np.array(self.window.get_queue())[:, column_index])
        return min

    def _get_max(self, column_index):
        max = 1.
        if not self.window.is_empty():
            max = np.nanmax(np.array(self.window.get_queue())[:, column_index])
        return max

    def partial_fit_transform(self, X, y=None):
        """ partial_fit_transform

        Partially fits the model and then apply the transform to the data.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The sample or set of samples that should be transformed.

        y: Array-like
            The true labels.

        Returns
        -------
        numpy.ndarray of shape (n_samples, n_features)
            The transformed data.

        """
        X = self.transform(X)

        return X

    def partial_fit(self, X, y=None):
        """ partial_fit

        Partial fits the model.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The sample or set of samples that should be transformed.

        y: Array-like
            The true labels.

        Returns
        -------
        MinmaxScaler
            self

        """
        X = np.asarray(X)
        self.window.add_element(X)
        return self
