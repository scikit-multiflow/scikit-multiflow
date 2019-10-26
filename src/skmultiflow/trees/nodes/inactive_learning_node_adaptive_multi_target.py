import numpy as np
from skmultiflow.trees.nodes import InactiveLearningNodePerceptronMultiTarget


class InactiveLearningNodeAdaptiveMultiTarget(InactiveLearningNodePerceptronMultiTarget):

    def __init__(self, initial_class_observations, perceptron_weight=None,
                 random_state=None):
        super().__init__(initial_class_observations, perceptron_weight,
                         random_state)

        # Faded errors for the perceptron and mean predictors
        self.fMAE_M = 0.0
        self.fMAE_P = 0.0

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
        rht: RegressionHoeffdingTree
            Regression Hoeffding Tree to update.
        """
        normalized_sample = rht.normalize_sample(X)
        normalized_pred = self.predict(normalized_sample)

        normalized_target_value = rht.normalized_target_value(y)
        self.perceptron_weight += learning_ratio * \
            np.matmul((normalized_target_value - normalized_pred)[:, None],
                      normalized_sample[None, :])

        self.normalize_perceptron_weights()

        # Update faded errors for the predictors
        # The considered errors are normalized, since they are based on
        # mean centered and sd scaled values
        self.fMAE_P = 0.95 * self.fMAE_P + np.abs(
            normalized_target_value - normalized_pred
        )

        self.fMAE_M = 0.95 * self.fMAE_M + np.abs(
            normalized_target_value - rht.
            normalized_target_value(self._observed_class_distribution[1] /
                                    self._observed_class_distribution[0])
        )
