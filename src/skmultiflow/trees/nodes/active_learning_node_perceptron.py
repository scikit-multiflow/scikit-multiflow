from copy import deepcopy

import numpy as np

from skmultiflow.trees.nodes import ActiveLearningNodeForRegression
from skmultiflow.trees.attribute_observer import NominalAttributeRegressionObserver
from skmultiflow.trees.attribute_observer import NumericAttributeRegressionObserver
from skmultiflow.utils import check_random_state


class ActiveLearningNodePerceptron(ActiveLearningNodeForRegression):
    """ Learning Node for regression tasks that always use a linear perceptron
    model to provide responses.

    Parameters
    ----------
    initial_class_observations: dict
        In regression tasks this dictionary carries the sufficient statistics
        to perform online variance calculation. They refer to the number of
        observations (key '0'), the sum of the target values (key '1'), and
        the sum of the squared target values (key '2').
    parent_node: ActiveLearningNodePerceptron (default=None)
        A node containing statistics about observed data.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_class_observations, parent_node=None, random_state=None):
        """ ActiveLearningNodePerceptron class constructor."""
        super().__init__(initial_class_observations)
        self.set_weight_seen_at_last_split_evaluation(self.get_weight_seen())
        self.random_state = check_random_state(random_state)
        self.samples_seen = 0
        if parent_node is None:
            self.perceptron_weight = None
        else:
            self.perceptron_weight = deepcopy(parent_node.perceptron_weight)
            self.samples_seen = parent_node.samples_seen

    def learn_from_instance(self, X, y, weight, rht):
        """Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: float
            Instance target value.
        weight: float
            Instance weight.
        rht: HoeffdingTreeRegressor
            Regression Hoeffding Tree to update.

        """

        # In regression, the self._observed_class_distribution dictionary keeps three statistics:
        # [0] sum of sample seen by the node
        # [1] sum of target values
        # [2] sum of squared target values
        # These statistics are useful to calculate the mean and to calculate the variance reduction

        if self.perceptron_weight is None:
            self.perceptron_weight = self.random_state.uniform(-1, 1, len(X)+1)

        try:
            self._observed_class_distribution[0] += weight
            self._observed_class_distribution[1] += y * weight
            self._observed_class_distribution[2] += y * y * weight
        except KeyError:
            self._observed_class_distribution[0] = weight
            self._observed_class_distribution[1] = y * weight
            self._observed_class_distribution[2] = y * y * weight

        # Update perceptron
        self.samples_seen = self._observed_class_distribution[0]

        if rht.learning_ratio_const:
            learning_ratio = rht.learning_ratio_perceptron
        else:
            learning_ratio = rht.learning_ratio_perceptron / \
                             (1 + self.samples_seen * rht.learning_ratio_decay)

        # Loop for compatibility with bagging methods
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
        y: float
            Instance target value.
        learning_ratio: float
            perceptron learning ratio
        rht: HoeffdingTreeRegressor
            Regression Hoeffding Tree to update.
        """
        normalized_sample = rht.normalize_sample(X)
        normalized_pred = np.dot(self.perceptron_weight, normalized_sample)
        normalized_target_value = rht.normalize_target_value(y)
        delta = normalized_target_value - normalized_pred
        self.perceptron_weight = self.perceptron_weight + learning_ratio * delta * normalized_sample
        # Normalize perceptron weights
        self.perceptron_weight = self.perceptron_weight / np.sum(self.perceptron_weight)


def compute_sd(square_val: float, val: float, size: float):
    if size > 1:
        a = square_val - ((val * val) / size)
        if a > 0:
            return np.sqrt(a / size)
    return 0.0
