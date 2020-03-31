import numpy as np

from skmultiflow.core import RegressorMixin
from skmultiflow.lazy.base_neighbors import BaseNeighbors
from skmultiflow.utils import get_dimensions, SlidingWindow


class KNNRegressor(BaseNeighbors, RegressorMixin):
    """k-Nearest Neighbors regressor.

    This non-parametric regression method keeps track of the last
    ``max_window_size`` training samples. Predictions are obtained by
    aggregating the values of the closest n_neighbors stored-samples with
    respect to a query sample.

    Parameters
    ----------
    n_neighbors: int (default=5)
        The number of nearest neighbors to search for.

    max_window_size: int (default=1000)
        The maximum size of the window storing the last observed samples.

    leaf_size: int (default=30)
        sklearn.KDTree parameter. The maximum number of samples that can
        be stored in one leaf node, which determines from which point the
        algorithm will switch for a brute-force approach. The bigger this
        number the faster the tree construction time, but the slower the
        query time will be.

    metric: string or sklearn.DistanceMetric object
        sklearn.KDTree parameter. The distance metric to use for the KDTree.
        Default=’euclidean’. KNNRegressor.valid_metrics() gives a list of
        the metrics which are valid for KDTree.

    aggregation_method: str (default='mean')
            | The method to aggregate the target values of neighbors.
            | 'mean'
            | 'median'

    Notes
    -----
    This estimator is not optimal for a mixture of categorical and numerical
    features.

    """

    _MEAN = 'mean'
    _MEDIAN = 'median'

    def __init__(self,
                 n_neighbors=5,
                 max_window_size=1000,
                 leaf_size=30,
                 metric='euclidean',
                 aggregation_method='mean'):

        super().__init__(n_neighbors=n_neighbors,
                         max_window_size=max_window_size,
                         leaf_size=leaf_size,
                         metric=metric)
        if aggregation_method not in {self._MEAN, self._MEDIAN}:
            raise ValueError("Invalid aggregation_method: {}.\n"
                             "Valid options are: {}".format(aggregation_method,
                                                            {self._MEAN, self._MEDIAN}))
        self.aggregation_method = aggregation_method

    def partial_fit(self, X, y, sample_weight=None):
        """ Partially fits the model on the samples X and targets y.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.

        y: numpy.ndarray of shape (n_samples)
            An array-like containing the target values for all
            samples in X.

        sample_weight: Not used.

        Returns
        -------
        KNNRegressor
            self

        Notes
        -----
        For the K-Nearest Neighbors regressor, fitting the model is the
        equivalent of inserting the newer samples in the observed window,
        and if the size_limit is reached, removing older results.

        """
        r, c = get_dimensions(X)

        for i in range(r):
            self.data_window.add_sample(X=X[i], y=y[i])
        return self

    def predict(self, X):
        """ Predict the target value for sample X

        Search the KDTree for the n_neighbors nearest neighbors.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the target value for.

        Returns
        -------
        np.ndarray
            An array containing the predicted target values for each \
            samples in X.

        """
        r, c = get_dimensions(X)
        predictions = np.zeros(r)
        for i in range(r):
            predictions[i] = self._predict(X)
        return predictions

    def _predict(self, X):
        r, c = get_dimensions(X)
        y_pred = np.zeros(r)
        if self.data_window is None or self.data_window.size < self.n_neighbors:
            # Not enough information available, return default predictions (0.0)
            return y_pred

        _, neighbors_idx = self._get_neighbors(X)
        neighbors_val = self.data_window.targets_buffer[neighbors_idx]
        if self.aggregation_method == self._MEAN:
            y_pred = np.mean(neighbors_val)
        else:   # self.aggregation_method == self._MEDIAN
            y_pred = np.median(neighbors_val)

        return y_pred

    def predict_proba(self, X):
        raise NotImplementedError('predict_proba is not implemented for this method.')
