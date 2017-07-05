__author__ = 'Guilherme Matsumoto'

from skmultiflow.classification.core.base_drift_detector import BaseDriftDetector


class PageHinkley(BaseDriftDetector):
    def __init__(self, min_num_instances=30, delta=0.005, _lambda=50, alpha=1-0.0001):
        super().__init__()
        self.min_instances = min_num_instances
        self.delta = delta
        self._lambda = _lambda
        self.alpha = alpha
        self.x_mean = None
        self.sample_count = None
        self.sum = None
        self.reset()

    def reset(self):
        super().reset()
        self.sample_count = 1
        self.x_mean = 0.0
        self.sum = 0.0

    def add_element(self, x):
        if self.in_concept_change:
            self.reset()

        self.x_mean = self.x_mean + (x - self.x_mean) / 1.0 * self.sample_count
        self.sum = self.alpha * self.sum + (x - self.x_mean - self.delta)

        self.sample_count += 1

        self.estimation = self.x_mean
        self.in_concept_change = False
        self.in_warning_zone = False

        self.delay = 0

        if (self.sample_count < self.min_instances):
            return None

        if self.sum > self._lambda:
            self.in_concept_change = True

    def get_info(self):
        return 'Not implemented.'
