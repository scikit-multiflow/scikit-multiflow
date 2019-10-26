import numpy as np
from skmultiflow.trees.nodes import InactiveLearningNode


class InactiveLearningNodePerceptron(InactiveLearningNode):

    def __init__(self, initial_class_observations, perceptron_weight=None):
        super().__init__(initial_class_observations)
        if perceptron_weight is None:
            self.perceptron_weight = []
        else:
            self.perceptron_weight = perceptron_weight

    def learn_from_instance(self, X, y, weight, rht):

        if self.perceptron_weight is None:
            self.perceptron_weight = np.random.uniform(-1, 1, len(X)+1)

        try:
            self._observed_class_distribution[0] += weight
        except KeyError:
            self._observed_class_distribution[0] = weight

        if rht.learning_ratio_const:
            learning_ratio = rht.learning_ratio_perceptron
        else:
            learning_ratio = rht.learning_ratio_perceptron / 1 + \
                self._observed_class_distribution[0] * rht.learning_ratio_decay

        try:
            self._observed_class_distribution[1] += y * weight
            self._observed_class_distribution[2] += y * y * weight
        except KeyError:
            self._observed_class_distribution[1] = y * weight
            self._observed_class_distribution[2] = y * y * weight

        for i in range(int(weight)):
            self.update_weights(X, y, learning_ratio, rht)

    def update_weights(self, X, y, learning_ratio, ht):
        normalized_sample = ht.normalize_sample(X)
        normalized_pred = self.predict(normalized_sample)
        normalized_target_value = ht.normalized_target_value(y)
        self.perceptron_weight += learning_ratio * \
            np.multiply((normalized_pred - normalized_target_value),
                        normalized_sample)

    def predict(self, X):
        return np.dot(self.perceptron_weight, X[0])
