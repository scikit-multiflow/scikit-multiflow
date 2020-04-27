import numpy as np

from skmultiflow.trees.nodes import InactiveLearningNodePerceptronMultiTarget
from skmultiflow.utils import get_dimensions


class SSTInactiveLearningNode(InactiveLearningNodePerceptronMultiTarget):
    """ Inactive Learning Node for SST-HT that always use stacked perceptrons to
    provide targets responses.

    Parameters
    ----------
    initial_class_observations: dict
        A dictionary containing the set of sufficient statistics to be
        stored by the leaf node. It contains the following elements:
        - 0: the sum of elements seen so far;
        - 1: the sum of the targets values seen so far;
        - 2: the sum of the squared values of the targets seen so far.
    parent_node: SSTActiveLearningNode (default=None)
        A node containing statistics about observed data.
    random_state : `int`, `RandomState` instance or None (default=None)
        If int, `random_state` is used as seed to the random number
        generator; If a `RandomState` instance, `random_state` is the
        random number generator; If `None`, the random number generator
        is the current `RandomState` instance used by `np.random`.
    """
    def __init__(self, initial_class_observations, parent_node=None,
                 random_state=None):
        """ SSTInactiveLearningNode class constructor."""
        super().__init__(initial_class_observations, parent_node,
                         random_state)

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
        self._observed_class_distribution[0] += weight

        if rht.learning_ratio_const:
            learning_ratio = rht.learning_ratio_perceptron
        else:
            learning_ratio = rht.learning_ratio_perceptron / \
                            (1 + self._observed_class_distribution[0] *
                             rht.learning_ratio_decay)

        self._observed_class_distribution[1] += weight * y
        self._observed_class_distribution[2] += weight * y * y

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
        normalized_base_pred = self._predict_base(normalized_sample)

        _, n_features = get_dimensions(X)
        _, n_targets = get_dimensions(y)

        normalized_target_value = rht.normalize_target_value(y)

        self.perceptron_weight[0] += learning_ratio * \
            (normalized_target_value - normalized_base_pred)[:, None] @ \
            normalized_sample[None, :]

        # Add bias term
        normalized_base_pred = np.append(normalized_base_pred, 1.0)

        normalized_meta_pred = self._predict_meta(normalized_base_pred)

        self.perceptron_weight[1] += learning_ratio * \
            (normalized_target_value - normalized_meta_pred)[:, None] @ \
            normalized_base_pred[None, :]

        self.normalize_perceptron_weights()

    # Normalize both levels
    def normalize_perceptron_weights(self):
        n_targets = self.perceptron_weight[0].shape[0]
        # Normalize perceptron weights
        for i in range(n_targets):
            sum_w_0 = np.sum(np.absolute(self.perceptron_weight[0][i, :]))
            self.perceptron_weight[0][i, :] /= sum_w_0
            sum_w_1 = np.sum(np.absolute(self.perceptron_weight[1][i, :]))
            self.perceptron_weight[1][i, :] /= sum_w_1

    def _predict_base(self, X):
        return self.perceptron_weight[0] @ X

    def _predict_meta(self, X):
        return self.perceptron_weight[1] @ X
