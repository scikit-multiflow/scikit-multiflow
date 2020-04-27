import math

from skmultiflow.trees.attribute_test import NominalAttributeMultiwayTest

from skmultiflow.trees.nodes import FoundNode
from skmultiflow.trees.nodes import SplitNode
from skmultiflow.trees.nodes import ActiveLearningNode
from skmultiflow.trees.nodes import InactiveLearningNode
from skmultiflow.trees.nodes import AdaNode
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.utils import check_random_state


class AdaSplitNodeForRegression(SplitNode, AdaNode):
    """ Node that splits the data in a Regression Hoeffding Adaptive Tree.

    Parameters
    ----------
    split_test: skmultiflow.split_test.InstanceConditionalTest
        Split test.
    class_observations: dict
        In regression tasks this dictionary carries the sufficient to perform
        online variance calculation. They refer to the number of observations
        (key '0'), the sum of the target values (key '1'), and the sum of the
        squared target values (key '2').
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, split_test, class_observations, random_state=None):
        super().__init__(split_test, class_observations)
        self._estimation_error_weight = ADWIN()
        self._alternate_tree = None
        self.error_change = False
        self.random_state = check_random_state(random_state)

        # To normalize the observed errors in the [0, 1] range
        self._min_error = float('Inf')
        self._max_error = float('-Inf')

    # Override AdaNode
    def number_leaves(self):
        num_of_leaves = 0
        for child in self._children.values():
            if child is not None:
                num_of_leaves += child.number_leaves()

        return num_of_leaves

    # Override AdaNode
    def get_error_estimation(self):
        return self._estimation_error_weight.estimation

    # Override AdaNode
    def get_error_width(self):
        w = 0.0
        if not self.is_null_error():
            w = self._estimation_error_weight.width

        return w

    # Override AdaNode
    def is_null_error(self):
        return self._estimation_error_weight is None

    # Override AdaNode
    def learn_from_instance(self, X, y, weight, rhat, parent, parent_branch):
        normalized_error = 0.0

        if self.filter_instance_to_leaf(X, parent, parent_branch).node is not None:
            y_pred = rhat.predict([X])[0]
            normalized_error = self.get_normalized_error(y, y_pred)
        if self._estimation_error_weight is None:
            self._estimation_error_weight = ADWIN()

        old_error = self.get_error_estimation()

        # Add element to Change detector
        self._estimation_error_weight.add_element(normalized_error)

        # Detect change
        self.error_change = self._estimation_error_weight.detected_change()

        if self.error_change and old_error > self.get_error_estimation():
            self.error_change = False

        # Check condition to build a new alternate tree
        if self.error_change:
            self._alternate_tree = rhat._new_learning_node()
            rhat.alternate_trees_cnt += 1

        # Condition to replace alternate tree
        elif self._alternate_tree is not None and not self._alternate_tree.is_null_error():
            if self.get_error_width() > rhat._ERROR_WIDTH_THRESHOLD \
                    and self._alternate_tree.get_error_width() > rhat._ERROR_WIDTH_THRESHOLD:
                old_error_rate = self.get_error_estimation()
                alt_error_rate = self._alternate_tree.get_error_estimation()
                fDelta = .05
                fN = 1.0 / self._alternate_tree.get_error_width() + 1.0 / self.get_error_width()

                bound = math.sqrt(2.0 * old_error_rate * (1.0 - old_error_rate) *
                                  math.log(2.0 / fDelta) * fN)
                # To check, bound never less than (old_error_rate - alt_error_rate)
                if bound < (old_error_rate - alt_error_rate):
                    rhat._active_leaf_node_cnt -= self.number_leaves()
                    rhat._active_leaf_node_cnt += self._alternate_tree.number_leaves()
                    self.kill_tree_children(rhat)

                    if parent is not None:
                        parent.set_child(parent_branch, self._alternate_tree)
                    else:
                        rhat._tree_root = rhat._tree_root._alternate_tree
                    rhat.switch_alternate_trees_cnt += 1
                elif bound < alt_error_rate - old_error_rate:
                    if isinstance(self._alternate_tree, ActiveLearningNode):
                        self._alternate_tree = None
                    elif isinstance(self._alternate_tree, InactiveLearningNode):
                        self._alternate_tree = None
                    else:
                        self._alternate_tree.kill_tree_children(rhat)
                    rhat.pruned_alternate_trees_cnt += 1  # hat.pruned_alternate_trees_cnt to check

        # Learn_From_Instance alternate Tree and Child nodes
        if self._alternate_tree is not None:
            self._alternate_tree.learn_from_instance(X, y, weight, rhat, parent, parent_branch)
        child_branch = self.instance_child_index(X)
        child = self.get_child(child_branch)
        if child is not None:
            child.learn_from_instance(X, y, weight, rhat, self, child_branch)
        # Instance contains a categorical value previously unseen by the split
        # node
        elif isinstance(self.get_split_test(), NominalAttributeMultiwayTest) and \
                self.get_split_test().branch_for_instance(X) < 0:
            # Creates a new learning node to encompass the new observed feature
            # value
            leaf_node = rhat._new_learning_node()
            branch_id = self.get_split_test().add_new_branch(
                X[self.get_split_test().get_atts_test_depends_on()[0]]
            )
            self.set_child(branch_id, leaf_node)
            rhat._active_leaf_node_cnt += 1
            leaf_node.learn_from_instance(X, y, weight, rhat, parent, parent_branch)

    # Override AdaNode
    def kill_tree_children(self, rhat):
        for child in self._children.values():
            if child is not None:
                # Delete alternate tree if it exists
                if isinstance(child, SplitNode) and child._alternate_tree is not None:
                    rhat.pruned_alternate_trees_cnt += 1
                # Recursive delete of SplitNodes
                if isinstance(child, SplitNode):
                    child.kill_tree_children(rhat)

                if isinstance(child, ActiveLearningNode):
                    child = None
                    rhat._active_leaf_node_cnt -= 1
                elif isinstance(child, InactiveLearningNode):
                    child = None
                    rhat._inactive_leaf_node_cnt -= 1

    # override AdaNode
    def filter_instance_to_leaves(self, X, y, weight, parent, parent_branch,
                                  update_splitter_counts=False, found_nodes=None):
        if found_nodes is None:
            found_nodes = []
        if update_splitter_counts:

            try:

                self._observed_class_distribution[0] += weight
                self._observed_class_distribution[1] += y * weight
                self._observed_class_distribution[2] += y * y * weight

            except KeyError:

                self._observed_class_distribution[0] = weight
                self._observed_class_distribution[1] = y * weight
                self._observed_class_distribution[2] = y * y * weight

        child_index = self.instance_child_index(X)
        if child_index >= 0:
            child = self.get_child(child_index)
            if child is not None:
                child.filter_instance_to_leaves(X, y, weight, parent, parent_branch,
                                                update_splitter_counts, found_nodes)
            else:
                found_nodes.append(FoundNode(None, self, child_index))
        if self._alternate_tree is not None:
            self._alternate_tree.filter_instance_to_leaves(X, y, weight, self, -999,
                                                           update_splitter_counts, found_nodes)

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
