import math
from skmultiflow.drift_detection import ADWIN
from skmultiflow.bayes import do_naive_bayes_prediction
from skmultiflow.trees.attribute_test import NominalAttributeMultiwayTest
from skmultiflow.trees.nodes import FoundNode
from skmultiflow.trees.nodes import SplitNode
from skmultiflow.trees.nodes import ActiveLearningNode
from skmultiflow.trees.nodes import ActiveLearningNodeNBAdaptive
from skmultiflow.trees.nodes import InactiveLearningNode
from skmultiflow.trees.nodes import AdaNode

from skmultiflow.utils import check_random_state, get_max_value_key, normalize_values_in_dict


class AdaLearningNode(ActiveLearningNodeNBAdaptive, AdaNode):
    """ Learning node for Hoeffding Adaptive Tree that uses Adaptive Naive
    Bayes models.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """
    def __init__(self, initial_stats, random_state=None):
        super().__init__(initial_stats)
        self._estimation_error_weight = ADWIN()
        self.error_change = False
        self._random_state = check_random_state(random_state)

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
    def learn_from_instance(self, X, y, weight, hat, parent, parent_branch):
        true_class = y

        if hat.bootstrap_sampling:
            # Perform bootstrap-sampling
            k = self._random_state.poisson(1.0)
            if k > 0:
                weight = weight * k

        class_prediction = get_max_value_key(self.get_class_votes(X, hat))

        bl_correct = (true_class == class_prediction)

        if self._estimation_error_weight is None:
            self._estimation_error_weight = ADWIN()

        old_error = self.get_error_estimation()

        # Add element to Adwin
        add = 0.0 if bl_correct else 1.0

        self._estimation_error_weight.add_element(add)
        # Detect change with Adwin
        self.error_change = self._estimation_error_weight.detected_change()

        if self.error_change and old_error > self.get_error_estimation():
            self.error_change = False

        # Update statistics
        super().learn_from_instance(X, y, weight, hat)

        # call ActiveLearningNode
        weight_seen = self.get_weight_seen()

        if weight_seen - self.get_weight_seen_at_last_split_evaluation() >= hat.grace_period:
            hat._attempt_to_split(self, parent, parent_branch)
            self.set_weight_seen_at_last_split_evaluation(weight_seen)

    # Override LearningNodeNBAdaptive
    def get_class_votes(self, X, ht):
        # dist = {}
        prediction_option = ht.leaf_prediction
        # MC
        if prediction_option == ht._MAJORITY_CLASS:
            dist = self.get_stats()
        # NB
        elif prediction_option == ht._NAIVE_BAYES:
            dist = do_naive_bayes_prediction(X, self._stats,
                                             self._attribute_observers)
        # NBAdaptive (default)
        else:
            if self._mc_correct_weight > self._nb_correct_weight:
                dist = self.get_stats()
            else:
                dist = do_naive_bayes_prediction(X, self._stats,
                                                 self._attribute_observers)

        dist_sum = sum(dist.values())  # sum all values in dictionary
        normalization_factor = dist_sum * self.get_error_estimation() * self.get_error_estimation()

        if normalization_factor > 0.0:
            dist = normalize_values_in_dict(dist, normalization_factor, inplace=False)

        return dist

    # Override AdaNode, New for option votes
    def filter_instance_to_leaves(self, X, y, weight, parent, parent_branch,
                                  update_splitter_counts, found_nodes=None):
        if found_nodes is None:
            found_nodes = []
        found_nodes.append(FoundNode(self, parent, parent_branch))


class AdaSplitNode(SplitNode, AdaNode):
    """ Node that splits the data in a Hoeffding Adaptive Tree.

    Parameters
    ----------
    split_test: skmultiflow.split_test.InstanceConditionalTest
        Split test.
    stats: dict (class_value, weight) or None
        Class observations
    """
    def __init__(self, split_test, stats, random_state=None):
        super().__init__(split_test, stats)
        self._estimation_error_weight = ADWIN()
        self._alternate_tree = None
        self.error_change = False

        self._random_state = check_random_state(random_state)

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
    def learn_from_instance(self, X, y, weight, hat, parent, parent_branch):
        true_class = y
        class_prediction = 0

        leaf = self.filter_instance_to_leaf(X, parent, parent_branch)
        if leaf.node is not None:
            class_prediction = get_max_value_key(leaf.node.get_class_votes(X, hat))

        bl_correct = (true_class == class_prediction)

        if self._estimation_error_weight is None:
            self._estimation_error_weight = ADWIN()

        old_error = self.get_error_estimation()

        # Add element to ADWIN
        add = 0.0 if bl_correct else 1.0

        self._estimation_error_weight.add_element(add)
        # Detect change with ADWIN
        self.error_change = self._estimation_error_weight.detected_change()

        if self.error_change and old_error > self.get_error_estimation():
            self.error_change = False

        # Check condition to build a new alternate tree
        if self.error_change:
            self._alternate_tree = hat._new_learning_node()
            hat.alternate_trees_cnt += 1

        # Condition to replace alternate tree
        elif self._alternate_tree is not None and not self._alternate_tree.is_null_error():
            if self.get_error_width() > hat._ERROR_WIDTH_THRESHOLD \
                    and self._alternate_tree.get_error_width() > hat._ERROR_WIDTH_THRESHOLD:
                old_error_rate = self.get_error_estimation()
                alt_error_rate = self._alternate_tree.get_error_estimation()
                fDelta = .05
                fN = 1.0 / self._alternate_tree.get_error_width() + 1.0 / self.get_error_width()

                bound = math.sqrt(2.0 * old_error_rate * (1.0 - old_error_rate) *
                                  math.log(2.0 / fDelta) * fN)
                # To check, bound never less than (old_error_rate - alt_error_rate)
                if bound < (old_error_rate - alt_error_rate):
                    hat._active_leaf_node_cnt -= self.number_leaves()
                    hat._active_leaf_node_cnt += self._alternate_tree.number_leaves()
                    self.kill_tree_children(hat)

                    if parent is not None:
                        parent.set_child(parent_branch, self._alternate_tree)
                    else:
                        # Switch tree root
                        hat._tree_root = hat._tree_root._alternate_tree
                    hat.switch_alternate_trees_cnt += 1
                elif bound < alt_error_rate - old_error_rate:
                    if isinstance(self._alternate_tree, ActiveLearningNode):
                        self._alternate_tree = None
                    elif isinstance(self._alternate_tree, InactiveLearningNode):
                        self._alternate_tree = None
                    else:
                        self._alternate_tree.kill_tree_children(hat)
                    hat.pruned_alternate_trees_cnt += 1  # hat.pruned_alternate_trees_cnt to check

        # Learn_From_Instance alternate Tree and Child nodes
        if self._alternate_tree is not None:
            self._alternate_tree.learn_from_instance(X, y, weight, hat, parent, parent_branch)
        child_branch = self.instance_child_index(X)
        child = self.get_child(child_branch)
        if child is not None:
            child.learn_from_instance(X, y, weight, hat, self, child_branch)
        # Instance contains a categorical value previously unseen by the split
        # node
        elif isinstance(self.get_split_test(), NominalAttributeMultiwayTest) and \
                self.get_split_test().branch_for_instance(X) < 0:
            # Creates a new learning node to encompass the new observed feature
            # value
            leaf_node = hat._new_learning_node()
            branch_id = self.get_split_test().add_new_branch(
                X[self.get_split_test().get_atts_test_depends_on()[0]]
            )
            self.set_child(branch_id, leaf_node)
            hat._active_leaf_node_cnt += 1
            leaf_node.learn_from_instance(X, y, weight, hat, parent, parent_branch)

    # Override AdaNode
    def kill_tree_children(self, hat):
        for child in self._children.values():
            if child is not None:
                # Delete alternate tree if it exists
                if isinstance(child, SplitNode) and child._alternate_tree is not None:
                    child._alternate_tree.kill_tree_children(hat)
                    hat.pruned_alternate_trees_cnt += 1
                # Recursive delete of SplitNodes
                if isinstance(child, SplitNode):
                    child.kill_tree_children(hat)

                if isinstance(child, ActiveLearningNode):
                    child = None
                    hat._active_leaf_node_cnt -= 1
                elif isinstance(child, InactiveLearningNode):
                    child = None
                    hat._inactive_leaf_node_cnt -= 1

    # override AdaNode
    def filter_instance_to_leaves(self, X, y, weight, parent, parent_branch,
                                  update_splitter_counts=False, found_nodes=None):
        if found_nodes is None:
            found_nodes = []
        if update_splitter_counts:
            try:
                self._stats[y] += weight  # Dictionary (class_value, weight)
            except KeyError:
                self._stats[y] = weight
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
