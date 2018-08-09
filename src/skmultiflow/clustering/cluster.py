from abc import ABCMeta, abstractmethod


class Cluster(metaclass=ABCMeta):

    def __init__(self):
        super().__init__()
        self.id = -1
        self.gt_label = -1
        self.measure_values = {}

    @abstractmethod
    def get_center(self):

        raise NotImplementedError

    @abstractmethod
    def get_weight(self):

        raise NotImplementedError

    @abstractmethod
    def get_inclusion_probability(self, X, weight):

        raise NotImplementedError

    @abstractmethod
    def sample(self, random_state):

        raise NotImplementedError

    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

    def is_ground_truth(self):
        return self.gt_label != -1

    def set_ground_truth(self, truth):
        self.gt_label = truth

    def get_ground_truth(self):
        return self.gt_label
