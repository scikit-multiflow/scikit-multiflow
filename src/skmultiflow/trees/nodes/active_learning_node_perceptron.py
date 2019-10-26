import numpy as np
from skmultiflow.trees.nodes import ActiveLearningNode
from skmultiflow.trees.attribute_observer import NominalAttributeRegressionObserver
from skmultiflow.trees.attribute_observer import NumericAttributeRegressionObserver
from skmultiflow.utils import check_random_state


class ActiveLearningNodePerceptron(ActiveLearningNode):

    def __init__(self, initial_class_observations, perceptron_weight=None, random_state=None):
        """
        ActiveLearningNodePerceptron class constructor
        Parameters
        ----------
        initial_class_observations
        perceptron_weight
        """
        super().__init__(initial_class_observations)
        if perceptron_weight is None:
            self.perceptron_weight = None
        else:
            self.perceptron_weight = perceptron_weight
        self.random_state = check_random_state(random_state)

    def learn_from_instance(self, X, y, weight, rht):
        """Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: int
            Instance class.
        weight: float
            Instance weight.
        rht: RegressionHoeffdingTree
            Regression Hoeffding Tree to update.

        """

        if self.perceptron_weight is None:
            self.perceptron_weight = self.random_state.uniform(-1, 1, len(X)+1)

        try:
            self._observed_class_distribution[0] += weight
        except KeyError:
            self._observed_class_distribution[0] = weight

        if rht.learning_ratio_const:
            learning_ratio = rht.learning_ratio_perceptron
        else:
            learning_ratio = rht.learning_ratio_perceptron / \
                             (1 + self._observed_class_distribution[0] * rht.learning_ratio_decay)

        try:
            self._observed_class_distribution[1] += y * weight
            self._observed_class_distribution[2] += y * y * weight
        except KeyError:
            self._observed_class_distribution[1] = y * weight
            self._observed_class_distribution[2] = y * y * weight

        for i in range(int(weight)):
            self.update_weights(X, y, learning_ratio, rht)

        for i in range(len(X)):
            try:
                obs = self._attribute_observers[i]
            except KeyError:
                if rht.nominal_attributes is not None and i in rht.nominal_attributes:
                    obs = NominalAttributeRegressionObserver()
                else:
                    obs = NumericAttributeRegressionObserver()
                self._attribute_observers[i] = obs
            obs.observe_attribute_class(X[i], y, weight)

    def update_weights(self, X, y, learning_ratio, rht):
        """
        Update the perceptron weights
        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: int
            Instance class.
        learning_ratio: float
            perceptron learning ratio
        rht: RegressionHoeffdingTree
            Regression Hoeffding Tree to update.
        """
        normalized_sample = rht.normalize_sample(X)
        normalized_pred = self.predict(normalized_sample)
        normalized_target_value = rht.normalized_target_value(y)
        self.perceptron_weight = self.perceptron_weight + learning_ratio * \
            np.multiply((normalized_target_value - normalized_pred),
                        normalized_sample)

    def predict(self, X):
        return np.dot(self.perceptron_weight, X)

    def get_weight_seen(self):
        """Calculate the total weight seen by the node.

        Returns
        -------
        float
            Total weight seen.

        """
        if self._observed_class_distribution == {}:
            return 0
        else:
            return self._observed_class_distribution[0]
