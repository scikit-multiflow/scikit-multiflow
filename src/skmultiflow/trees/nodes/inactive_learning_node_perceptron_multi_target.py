import numpy as np
from skmultiflow.trees.nodes import InactiveLearningNodePerceptron
from skmultiflow.utils import check_random_state
from skmultiflow.utils import get_dimensions


class InactiveLearningNodePerceptronMultiTarget(InactiveLearningNodePerceptron):
    """ Inactive Learning Node for Multi-target Regression tasks that always use
    linear perceptron predictors for each target.

    Parameters
    ----------
    initial_class_observations: dict
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the targets values (key '1'), and the sum of the
        squared targets values (key '2').
    parent_node: ActiveLearningNodePerceptronMultiTarget (default=None)
        A node containing statistics about observed data.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_class_observations, parent_node=None,
                 random_state=None):
        """ InactiveLearningNodeForRegression class constructor."""
        super().__init__(initial_class_observations)

        if parent_node is None:
            self.perceptron_weight = None
        else:
            self.perceptron_weight = parent_node.perceptron_weight
        self.random_state = check_random_state(random_state)

    def learn_from_instance(self, X, y, weight, rht):
        """Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: numpy.ndarray of length equal to the number of targets.
            Instance targets.
        weight: float
            Instance weight.
        rht: HoeffdingTreeRegressor
            Regression Hoeffding Tree to update.
        """
        if self.perceptron_weight is None:
            # Creates matrix of perceptron random weights
            _, rows = get_dimensions(y)
            _, cols = get_dimensions(X)

            self.perceptron_weight = self.random_state.uniform(-1, 1,
                                                               (rows,
                                                                cols + 1))
            self.normalize_perceptron_weights()

        try:
            self._observed_class_distribution[0] += weight
        except KeyError:
            self._observed_class_distribution[0] = weight

        if rht.learning_ratio_const:
            learning_ratio = rht.learning_ratio_perceptron
        else:
            learning_ratio = rht.learning_ratio_perceptron / \
                            (1 + self._observed_class_distribution[0] *
                             rht.learning_ratio_decay)

        try:
            self._observed_class_distribution[1] += weight * y
            self._observed_class_distribution[2] += weight * y * y
        except KeyError:
            self._observed_class_distribution[1] = weight * y
            self._observed_class_distribution[2] = weight * y * y

        for i in range(int(weight)):
            self.update_weights(X, y, learning_ratio, rht)

    def update_weights(self, X, y, learning_ratio, rht):
        """Update the perceptron weights

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: numpy.ndarray of length equal to the number of targets.
            Targets values.
        learning_ratio: float
            perceptron learning ratio
        rht: HoeffdingTreeRegressor
            Regression Hoeffding Tree to update.
        """
        normalized_sample = rht.normalize_sample(X)
        normalized_pred = self.predict(normalized_sample)

        normalized_target_value = rht.normalize_target_value(y)
        self.perceptron_weight += learning_ratio * \
            np.matmul((normalized_target_value - normalized_pred)[:, None],
                      normalized_sample[None, :])

        self.normalize_perceptron_weights()

    def normalize_perceptron_weights(self):
        n_targets = self.perceptron_weight.shape[0]
        # Normalize perceptron weights
        for i in range(n_targets):
            sum_w = np.sum(np.abs(self.perceptron_weight[i, :]))
            self.perceptron_weight[i, :] /= sum_w

    # Predicts new income instances as a multiplication of the neurons
    # weights with the inputs augmented with a bias value
    def predict(self, X):
        return np.matmul(self.perceptron_weight, X)
