from skmultiflow.clustering.cluster import Cluster
from skmultiflow.utils import check_random_state
import numpy as np


class SphereCluster(Cluster):

    def __init__(self, center=None, radius=None, weighted_size=None, random_state=None, dimensions=None):
        super().__init__()

        if random_state is None and dimensions is None:
            self.center = center
            self.radius = radius
            self.weight = weighted_size
        elif center is None and weighted_size is None:
            self.center = [None] * dimensions
            self.radius = radius
            self._original_random_state = random_state
            self.random_state = check_random_state(self._original_random_state)
            interval = 1.0 - 2 * radius
            for i in range(len(self.center)):
                self.center[i] = (self.random_state.rand() * interval) + radius
            self.weight = 0.0

    def get_center(self):
        copy = self.center.copy()
        return copy

    def set_center(self, center):
        self.center = center

    def get_radius(self):
        return self.radius

    def set_radius(self, radius):
        self.radius = radius

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight

    def get_center_distance(self, X, weight):
        distance = 0.0
        center = self.get_center()
        for i in range(len(center)):
            d = center[i] - X[i]
            distance += d * d
        return np.sqrt(distance)

    def get_inclusion_probability(self, X, weight):
        if self.get_center_distance(X, weight) <= self.get_radius():
            return 1.0

        return 0.0

    def sample(self, random_state):
        raise NotImplementedError