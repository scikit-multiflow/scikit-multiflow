from skmultiflow.core.base import StreamModel
from skmultiflow.utils.utils import *
from skmultiflow.clustering.clustream.clustream_kernel import ClustreamKernel

import sys
import numpy as np
from sklearn.cluster import KMeans


class Clustream(StreamModel):
    """Clustream

    It maintains statistical information about the data using micro-clusters.
    These micro-clusters are temporal extensions of cluster feature vectors.
    The micro-clusters are stored at snapshots in time following a pyramidal
    pattern. This pattern allows to recall summary statistics from different
    time horizons in [1]_.

    Parameters
    ----------

    time_window: int (Default : 1000)
      The rang of the window
      if the current time is T and the time window is h, we should only consider
      about the data that arrived within the period (T-h,T)

    max_kernels: int (Default: 100)
      The Maximum number of micro kernels to use

    kernel_radius_factor: int (Default: 2)
       Multiplier for the kernel radius

    number_of_clusters: int (Default : 5)
        the clusters returned by the Kmeans algorithm using the summaries statistics


    References
    ----------
    .. [1] A. Kumar , A. Singh, and R. Singh. An efficient hybrid-clustream algorithm
       for stream mining

    """
    def __init__(self, time_window=1000, max_kernels=100, kernel_radius_factor=2, number_of_clusters=5):
        super().__init__()
        self.time_window = time_window
        self.time_stamp = -1
        self.kernels = [None]*max_kernels
        self.initialized = False
        self.buffer = []
        self.buffer_size = max_kernels
        self.T = kernel_radius_factor
        self.M = max_kernels
        self.k = number_of_clusters
        self._train_weight_seen_by_model = 0.0

    def partial_fit(self, X, weight=None):
        """Incrementally trains the model. Train samples (instances) are composed of X attributes .

        Tasks performed before training:

        * Verify instance weight. if not provided, uniform weights (1.0) are assumed.
        * If more than one instance is passed, loop through X and pass instances one at a time.
        * Update weight seen by model.

        Training tasks:

        * determinate closest kernel
        * Check whether instance fits into closest Kernel:
            1- if data fits, put into kernel
            2- if data does not fit , we need to free some space to insert a new kernel
            and this can be done in two ways , delete an old kernel or merge two kernels
            which are close to each other

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Inst    ance attributes.

        weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.

        """

        if weight is None:
            weight = np.array([1.0])
        row_cnt, _ = get_dimensions(X)
        weight_row_cnt, _ = get_dimensions(weight)
        if row_cnt != weight_row_cnt:
            weight = [weight[0]] * row_cnt
        for i in range(row_cnt):
            if weight[i] != 0.0:
                self._train_weight_seen_by_model += weight[i]
                self._partial_fit(X[i], weight[i])

    def _partial_fit(self, X, weight):

        dim = len(X)
        self.time_stamp += 1

        if not self.initialized:
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(ClustreamKernel(X, weight, dim, self.time_stamp, self.T, self.M))
                return
            else:
                for i in range(self.buffer_size):
                    self.kernels[i] = ClustreamKernel(X=self.buffer[i].get_center(), weight=1.0, dimensions=dim,
                                                      timestamp=self.time_stamp, T=self.T, M=self.M)
            self.buffer.clear()
            self.initialized = True

            return

        """determinate closest kernel"""
        closest_kernel = None
        min_distance = sys.float_info.max  # 1.7976931348623157e+308  maximum value of float
        for i in range(len(self.kernels)):
            distance = self._distance(X, self.kernels[i].get_center())
            if distance < min_distance:
                closest_kernel = self.kernels[i]
                min_distance = distance

        """check whether the instance fits into closest kernel"""

        radius = 0.0
        if closest_kernel.get_weight() == 1:
            radius = sys.float_info.max
            center = closest_kernel.get_center()
            for i in range(len(self.kernels)):
                if self.kernels[i] == closest_kernel:
                    continue
                distance = self._distance(self.kernels[i].get_center(), center)
                radius = min(distance, radius)
        else:
            radius = closest_kernel.get_radius()

        if min_distance < radius:
            closest_kernel.insert(X, weight, self.time_stamp)
            return

        """Data does not fit , we need to free some space in order to insert a new kernel"""

        threshold = self.time_stamp - self.time_window

        """try to delete old kernel"""

        for i in range(len(self.kernels)):
            if self.kernels[i].get_relevance_stamp() < threshold:

                self.kernels[i] = ClustreamKernel(X=X, weight=weight, dimensions=dim, timestamp=self.time_stamp,
                                                  T=self.T, M=self.M)

                return

        """try to merge closest two kernels"""

        closest_a = 0
        closest_b = 0
        min_distance = sys.float_info.max
        for i in range(len(self.kernels)):
            center_a = self.kernels[i].get_center()
            for j in range(i+1, len(self.kernels)):
                dist = self._distance(center_a, self.kernels[j].get_center())
                if dist < min_distance:
                    min_distance = dist
                    closest_a = i
                    closest_b = j
        assert closest_a != closest_b
        self.kernels[closest_a].add(self.kernels[closest_b])
        self.kernels[closest_b] = ClustreamKernel(X=X, weight=weight, dimensions=dim, timestamp=self.time_stamp,
                                                  T=self.T, M=self.M)

    def get_micro_clustering_result(self):

        if not self.initialized:
            return []
        res = [None]*len(self.kernels)
        for i in range(len(res)):
            res[i] = ClustreamKernel(cluster=self.kernels[i], T=self.T, M=self.M)
        return res

    def get_clustering_result(self):

        if not self.initialized:
            return []
        micro_cluster_centers = np.array([micro_cluster.get_center() for
                                          micro_cluster in self.get_micro_clustering_result()])

        kmeans = KMeans(n_clusters=self.k).fit(micro_cluster_centers)
        return kmeans

    def fit_predict(self, X, weight=None):
        """
        Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling partial_fit(X) followed by predict(X).

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Inst    ance attributes.

        weight: float or array-like
            Instance weight. If not provided, uniform weights are assumed.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Cluster labels
        """

        if weight is None:
            weight = np.array([1.0])
        row_cnt, _ = get_dimensions(X)
        weight_row_cnt, _ = get_dimensions(weight)
        if row_cnt != weight_row_cnt:
            weight = [weight[0]] * row_cnt
        for i in range(row_cnt):
            if weight[i] != 0.0:
                self._train_weight_seen_by_model += weight[i]
                self._partial_fit(X[i], weight[i])

        micro_cluster_centers = np.array([micro_cluster.get_center() for
                                          micro_cluster in self.get_micro_clustering_result()])

        kmeans = KMeans(n_clusters=self.k).fit(micro_cluster_centers)

        y = []
        for i in range(len(X)):
            index, _ = self._get_closest_kernel(X[i], micro_cluster_centers)

            y.append(kmeans.labels_[index])

        return y

    @staticmethod
    def _get_closest_kernel(X, micro_clusters):
        min_distance = sys.float_info.max
        closest_kernel = None
        closest_kernel_index = -1
        for i, micro_cluster in enumerate(micro_clusters):
            distance = np.linalg.norm(micro_cluster.get_center() - X)
            if distance < min_distance:
                min_distance = distance
                closest_kernel = micro_cluster
                closest_kernel_index = i
        return closest_kernel_index, min_distance

    @staticmethod
    def implements_micro_clustering():
        return True

    def get_name(self):
        return "clustream" + str(self.time_window)

    @staticmethod
    def _distance(point_a, point_b):
        distance = 0.0
        for i in range(len(point_a)):
            d = point_a[i] - point_b[i]
            distance += d * d
        return np.sqrt(distance)

    def predict(self, X):

        micro_cluster_centers = np.array([micro_cluster.get_center() for
                                          micro_cluster in self.get_micro_clustering_result()])

        kmeans = KMeans(n_clusters=self.k).fit(micro_cluster_centers)

        y = []
        for i in range(len(X)):
            index, _ = self._get_closest_kernel(X[i], self.get_micro_clustering_result())

            y.append(kmeans.labels_[index])

        return y

    def predict_proba(self, X):
        pass

    def reset(self):
        pass

    def score(self, X, y):
        pass

    def fit(self, X, y, classes=None, weight=None):
        pass

    def get_info(self):
        pass





















