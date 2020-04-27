from skmultiflow.trees.nodes import FoundNode
from skmultiflow.trees.nodes import ActiveLearningNodePerceptron
from skmultiflow.trees.nodes import AdaNode
from skmultiflow.drift_detection.adwin import ADWIN


class AdaLearningNodeForRegression(ActiveLearningNodePerceptron, AdaNode):
    """ Learning Node of the Regression Hoeffding Adaptive Tree that always use
    a linear perceptron model to provide responses.

    Parameters
    ----------
    initial_class_observations: dict
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the target values (key '1'), and the sum of the
        squared target values (key '2').
    parent_node: AdaLearningNodeForRegression (default=None)
        A node containing statistics about observed data.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, initial_class_observations, parent_node, random_state=None):
        super().__init__(initial_class_observations, parent_node, random_state)
        self._estimation_error_weight = ADWIN()
        self._error_change = False

        # To normalize the observed errors in the [0, 1] range
        self._min_error = float('Inf')
        self._max_error = float('-Inf')

    # Override AdaNode
    def number_leaves(self):
        return 1

    # Override AdaNode
    def get_error_estimation(self):
        return self._estimation_error_weight.estimation

    # Override AdaNode
    def get_error_width(self):
        return self._estimation_error_weight.width

    # Override AdaNode
    def is_null_error(self):
        return self._estimation_error_weight is None

    def kill_tree_children(self, hat):
        pass

    # Override AdaNode
    def learn_from_instance(self, X, y, weight, rhat, parent, parent_branch):

        super().learn_from_instance(X, y, weight, rhat)

        y_pred = rhat.predict([X])[0]
        normalized_error = self.get_normalized_error(y, y_pred)

        if self._estimation_error_weight is None:
            self._estimation_error_weight = ADWIN()

        old_error = self.get_error_estimation()

        # Add element to Adwin

        self._estimation_error_weight.add_element(normalized_error)
        # Detect change with Adwin
        self._error_change = self._estimation_error_weight.detected_change()

        if self._error_change and old_error > self.get_error_estimation():
            self._error_change = False

        # call ActiveLearningNode
        weight_seen = self.get_weight_seen()

        if weight_seen - self.get_weight_seen_at_last_split_evaluation() >= rhat.grace_period:
            rhat._attempt_to_split(self, parent, parent_branch)
            self.set_weight_seen_at_last_split_evaluation(weight_seen)

    # Override AdaNode, New for option votes
    def filter_instance_to_leaves(self, X, y, weight, parent, parent_branch,
                                  update_splitter_counts, found_nodes=None):
        if found_nodes is None:
            found_nodes = []
        found_nodes.append(FoundNode(self, parent, parent_branch))

    def get_normalized_error(self, y, y_pred):
        abs_error = abs(y - y_pred)

        # Incremental maintenance of the normalization ranges
        if abs_error < self._min_error:
            self._min_error = abs_error
        if abs_error > self._max_error:
            self._max_error = abs_error

        if self._min_error != self._max_error:
            return (abs_error - self._min_error) / (self._max_error - self._min_error)
        else:
            return 0.0
