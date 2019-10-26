from skmultiflow.trees.nodes import InactiveLearningNode


class InactiveLearningNodeForRegression(InactiveLearningNode):

    def __init__(self, initial_class_observations=None):
        """ InactiveLearningNode class constructor. """
        super().__init__(initial_class_observations)

    def learn_from_instance(self, X, y, weight, ht):

        try:
            self._observed_class_distribution[0] += weight
            self._observed_class_distribution[1] += y * weight
            self._observed_class_distribution[2] += y * y * weight
        except KeyError:
            self._observed_class_distribution[0] = weight
            self._observed_class_distribution[1] = y * weight
            self._observed_class_distribution[2] = y * y * weight
