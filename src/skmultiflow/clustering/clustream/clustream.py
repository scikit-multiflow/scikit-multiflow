from skmultiflow.core.base import StreamModel
from skmultiflow.utils.utils import *
from skmultiflow.clustering.clustream.clustream_kernel import ClustreamKernel
from skmultiflow.clustering.sphere_cluster import SphereCluster
import sys
import numpy as np


class Clustream(StreamModel):

    def __init__(self, time_window=1000, max_kernels=100, kernel_radius_factor=2):
        super().__init__()
        self.time_window = time_window
        self.time_stamp = -1
        self.kernels = [None]*max_kernels
        self.initialized = False
        self.buffer = []
        self.buffer_size = max_kernels
        self.T = kernel_radius_factor
        self.M = max_kernels
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
            Instance attributes.

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
            temp = len(self.kernels)
            assert temp <= self.buffer_size
            centers = [None]*temp
            for i in range(temp):
                centers[i] = self.buffer[i]
            kmeans_clustering = self.kMeans(temp, centers, self.buffer)

            for i in range(len(kmeans_clustering)):
                self.kernels[i] = ClustreamKernel(X=centers[i].get_center(), weight=1.0, dimensions=dim,
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

    def kMeans(self, k, centers, data):

        assert (k > 0)
        assert len(centers) == k

        dimensions = len(centers[0].get_center())
        clustering = [[] for _ in range(k)]
        repetitions = 100

        while repetitions - 1 >= 0:
            repetitions -= 1
            for _, point in enumerate(data):
                min_distance = self._distance(point.get_center(), centers[0].get_center())
                closest_cluster = 0
                for i in range(1, k):
                    distance = self._distance(point.get_center(), centers[i].get_center())
                    if distance < min_distance:
                        closest_cluster = i
                        min_distance = distance
                clustering[closest_cluster].append(point)
            new_centers = [None]*len(centers)
            for i in range(k):
                new_centers[i] = self._calculate_center(clustering[i], dimensions)
                clustering[i].clear()
            centers = new_centers

        return centers

    def _calculate_center(self, clusters, dimensions):

        res = [0.0]*dimensions
        if len(clusters) == 0:
            return SphereCluster(center=res, radius=0.0, weighted_size=1.0)
        for _, cluster in enumerate(clusters):
            center = cluster.get_center()
            for i in range(len(res)):
                res[i] += center[i]
        """normalize"""
        for i in range(len(res)):
            res[i] /= len(clusters)
        """calculate radius"""
        radius = 0.0
        for _, cluster in enumerate(clusters):
            dist = self._distance(res, cluster.get_center())
            if dist > radius:
                radius = dist

        sphere_cluster = SphereCluster(center=res, radius=radius, weighted_size=len(clusters))
        return sphere_cluster

    @staticmethod
    def implements_micro_clustering(self):
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

    def fit(self, X, y, classes=None, weight=None):
        pass

    def get_info(self):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def reset(self):
        pass

    def score(self, X, y):
        pass























