from operator import attrgetter

import numpy as np

from skmultiflow.core import MultiOutputMixin
from skmultiflow.trees import iSOUPTreeRegressor
from skmultiflow.utils import get_dimensions
from skmultiflow.trees.split_criterion import IntraClusterVarianceReductionSplitCriterion

from skmultiflow.trees.nodes import LearningNode
from skmultiflow.trees.nodes import SSTActiveLearningNode
from skmultiflow.trees.nodes import SSTActiveLearningNodeAdaptive
from skmultiflow.trees.nodes import SSTInactiveLearningNode
from skmultiflow.trees.nodes import SSTInactiveLearningNodeAdaptive


class StackedSingleTargetHoeffdingTreeRegressor(iSOUPTreeRegressor, MultiOutputMixin):
    """Stacked Single-target Hoeffding Tree regressor.

    Implementation of the Stacked Single-target Hoeffding Tree (SST-HT) method
    for multi-target regression as proposed by S. M. Mastelini, S. Barbon Jr.,
    and A. C. P. L. F. de Carvalho [1]_.

    Parameters
    ----------
    max_byte_size: int (default=33554432)
        Maximum memory consumed by the tree.
    memory_estimate_period: int (default=1000000)
        Number of instances between memory consumption checks.
    grace_period: int (default=200)
        Number of instances a leaf should observe between split attempts.
    split_confidence: float (default=0.0000001)
        Allowed error in split decision, a value closer to 0 takes longer to
        decide.
    tie_threshold: float (default=0.05)
        Threshold below which a split will be forced to break ties.
    binary_split: boolean (default=False)
        If True, only allow binary splits.
    stop_mem_management: boolean (default=False)
        If True, stop growing as soon as memory limit is hit.
    remove_poor_atts: boolean (default=False)
        If True, disable poor attributes.
    no_preprune: boolean (default=False)
        If True, disable pre-pruning.
    leaf_prediction: string (default='perceptron')
        | Prediction mechanism used at leafs.
        | 'perceptron' - Stacked perceptron
        | 'adaptive' - Adaptively chooses between the best predictor (mean,
          perceptron or stacked perceptron)
    nb_threshold: int (default=0)
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes
        are numerical.
    learning_ratio_perceptron: float
        The learning rate of the perceptron.
    learning_ratio_decay: float
        Decay multiplier for the learning rate of the perceptron
    learning_ratio_const: Bool
        If False the learning ratio will decay with the number of examples seen
    random_state: int, RandomState instance or None, optional (default=None)
       If int, random_state is the seed used by the random number generator;
       If RandomState instance, random_state is the random number generator;
       If None, the random number generator is the RandomState instance used
       by `np.random`. Used when leaf_prediction is 'perceptron'.

    References
    ----------
    .. [1] Mastelini, S. M., Barbon Jr, S., de Carvalho, A. C. P. L. F. (2019).
       "Online Multi-target regression trees with stacked leaf models". arXiv
       preprint arXiv:1903.12483.

    Examples
    --------
    .. code-block:: python

       # Imports
       from skmultiflow.data import RegressionGenerator
       from skmultiflow.trees import StackedSingleTargetHoeffdingTreeRegressor
       import numpy as np

       # Setup a data stream
       n_targets = 3
       stream = RegressionGenerator(n_targets=n_targets, random_state=1)

       # Setup the Stacked Single-target Hoeffding Tree Regressor
       sst_ht = StackedSingleTargetHoeffdingTreeRegressor()

       # Auxiliary variables to control loop and track performance
       n_samples = 0
       correct_cnt = 0
       max_samples = 200
       y_pred = np.zeros((max_samples, n_targets))
       y_true = np.zeros((max_samples, n_targets))

       # Run test-then-train loop for max_samples or while there is data in the stream
       while n_samples < max_samples and stream.has_more_samples():
           X, y = stream.next_sample()
           y_true[n_samples] = y[0]
           y_pred[n_samples] = sst_ht.predict(X)[0]
           sst_ht.partial_fit(X, y)
           n_samples += 1

       # Display results
       print('{} samples analyzed.'.format(n_samples))
       print('Stacked Single-target Hoeffding Tree Regressor mean absolute error: {}'.format(np.mean(np.abs(y_true - y_pred))))
    """

    # =====================================================================
    # == Stacked Single-target Hoeffding Regression Tree implementation ===
    # =====================================================================

    def __init__(self,
                 max_byte_size=33554432,
                 memory_estimate_period=1000000,
                 grace_period=200,
                 split_confidence=0.0000001,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_atts=False,
                 leaf_prediction='perceptron',
                 no_preprune=False,
                 nb_threshold=0,
                 nominal_attributes=None,
                 learning_ratio_perceptron=0.02,
                 learning_ratio_decay=0.001,
                 learning_ratio_const=True,
                 random_state=None):
        super().__init__(max_byte_size=max_byte_size,
                         memory_estimate_period=memory_estimate_period,
                         grace_period=grace_period,
                         split_confidence=split_confidence,
                         tie_threshold=tie_threshold,
                         binary_split=binary_split,
                         stop_mem_management=stop_mem_management,
                         remove_poor_atts=remove_poor_atts,
                         no_preprune=no_preprune,
                         leaf_prediction=leaf_prediction,
                         nb_threshold=nb_threshold,
                         nominal_attributes=nominal_attributes)
        self.split_criterion = 'icvr'   # intra cluster variance reduction
        self.learning_ratio_perceptron = learning_ratio_perceptron
        self.learning_ratio_decay = learning_ratio_decay
        self.learning_ratio_const = learning_ratio_const
        self.random_state = random_state

        self._tree_root = None
        self._decision_node_cnt = 0
        self._active_leaf_node_cnt = 0
        self._inactive_leaf_node_cnt = 0
        self._inactive_leaf_byte_size_estimate = 0.0
        self._active_leaf_byte_size_estimate = 0.0
        self._byte_size_estimate_overhead_fraction = 1.0
        self._growth_allowed = True
        self._train_weight_seen_by_model = 0.0

        self.examples_seen = 0
        self.sum_of_values = 0.0
        self.sum_of_squares = 0.0
        self.sum_of_attribute_values = 0.0
        self.sum_of_attribute_squares = 0.0

        # To add the n_targets property once
        self._n_targets_set = False

    @property
    def leaf_prediction(self):
        return self._leaf_prediction

    @leaf_prediction.setter
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction not in {self._PERCEPTRON, self._ADAPTIVE}:
            print("Invalid leaf_prediction option {}', will use default '{}'".
                  format(leaf_prediction, self._PERCEPTRON))
            self._leaf_prediction = self._PERCEPTRON
        else:
            self._leaf_prediction = leaf_prediction

    def _get_predictors_faded_error(self, X):
        """Get the faded error of the leaf corresponding to the pased instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        Returns
        -------
        dict (predictor, fmae)
        """
        fmaes = {}
        if self._tree_root is not None:
            found_node = self._tree_root.filter_instance_to_leaf(X, None, -1)
            leaf_node = found_node.node
            if leaf_node is None:
                leaf_node = found_node.parent
            if isinstance(leaf_node, LearningNode):
                fmaes['mean'] = leaf_node.fMAE_M
                fmaes['perceptron'] = leaf_node.fMAE_P
                fmaes['stacked_perceptron'] = leaf_node.fMAE_SP
            else:
                # If the found node is not a learning node, give preference to
                # the mean predictor
                fmaes['mean'] = np.zeros(self._n_targets)
                fmaes['perceptron'] = np.full(self._n_targets, np.Inf)
                fmaes['stacked_perceptron'] = np.full(self._n_targets, np.Inf)

        return fmaes

    def _new_learning_node(self, initial_class_observations=None, parent_node=None,
                           is_active_node=True):
        """Create a new learning node. The type of learning node depends on
        the tree configuration.
        """
        if initial_class_observations is None:
            initial_class_observations = {}

        if is_active_node:
            if self.leaf_prediction == self._PERCEPTRON:
                return SSTActiveLearningNode(
                    initial_class_observations,
                    parent_node,
                    random_state=self.random_state
                )
            elif self.leaf_prediction == self._ADAPTIVE:
                new_node = SSTActiveLearningNodeAdaptive(
                    initial_class_observations,
                    parent_node,
                    random_state=self.random_state
                )
                # Resets faded errors
                new_node.fMAE_M = np.zeros(self._n_targets, dtype=np.float64)
                new_node.fMAE_P = np.zeros(self._n_targets, dtype=np.float64)
                new_node.fMAE_SP = np.zeros(self._n_targets, dtype=np.float64)
                return new_node
        else:
            if self.leaf_prediction == self._PERCEPTRON:
                return SSTInactiveLearningNode(
                    initial_class_observations,
                    parent_node,
                    random_state=parent_node.random_state
                )
            elif self.leaf_prediction == self._ADAPTIVE:
                new_node = SSTInactiveLearningNodeAdaptive(
                    initial_class_observations,
                    parent_node,
                    random_state=parent_node.random_state
                )
                new_node.fMAE_M = parent_node.fMAE_M
                new_node.fMAE_P = parent_node.fMAE_P
                new_node.fMAE_SP = parent_node.fMAE_SP
                return new_node

    def predict(self, X):
        """Predicts the target value using mean class or the perceptron.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        list
            Predicted target values.
        """
        r, _ = get_dimensions(X)

        try:
            predictions = np.zeros((r, self._n_targets), dtype=np.float64)
        except AttributeError:
            return [0.0]
        for i in range(r):
            if self.leaf_prediction == self._PERCEPTRON:
                if self.examples_seen > 1:
                    perceptron_weights = self.get_weights_for_instance(X[i])
                    if perceptron_weights is None:
                        # Instance was sorted to a non-learning node: use
                        # mean prediction
                        votes = self.get_votes_for_instance(X[i]).copy()
                        number_of_examples_seen = votes[0]
                        sum_of_values = votes[1]
                        predictions[i] = sum_of_values / number_of_examples_seen
                        continue

                    normalized_sample = self.normalize_sample(X[i])
                    normalized_base_prediction = np.matmul(
                        perceptron_weights[0], normalized_sample
                    )
                    normalized_meta_prediction = np.matmul(
                        perceptron_weights[1],
                        np.append(normalized_base_prediction, 1.0)
                    )
                    mean = self.sum_of_values / self.examples_seen
                    variance = (self.sum_of_squares -
                                (self.sum_of_values *
                                 self.sum_of_values) /
                                self.examples_seen) / (self.examples_seen - 1)
                    sd = np.sqrt(variance, out=np.zeros_like(variance),
                                 where=variance >= 0.0)
                    # Samples are normalized using just one sd, as proposed in
                    # the iSoup-Tree method
                    predictions[i] = normalized_meta_prediction * sd + mean
            elif self.leaf_prediction == self._ADAPTIVE:
                if self.examples_seen > 1:
                    # Mean predictor
                    votes = self.get_votes_for_instance(X[i]).copy()
                    number_of_examples_seen = votes[0]
                    sum_of_values = votes[1]
                    pred_M = sum_of_values / number_of_examples_seen

                    # Perceptron variants
                    perceptron_weights = self.get_weights_for_instance(X[i])
                    if perceptron_weights is None:
                        # Instance was sorted to a non-learning node: use
                        # mean prediction
                        predictions[i] = pred_M
                        continue
                    else:
                        normalized_sample = self.normalize_sample(X[i])

                        # Standard perceptron
                        normalized_base_prediction = np.matmul(
                            perceptron_weights[0], normalized_sample
                        )
                        # Stacked perceptron
                        normalized_meta_prediction = np.matmul(
                            perceptron_weights[1],
                            np.append(normalized_base_prediction, 1.0)
                        )

                        mean = self.sum_of_values / self.examples_seen
                        variance = (self.sum_of_squares -
                                    (self.sum_of_values *
                                     self.sum_of_values) /
                                    self.examples_seen) / (self.examples_seen - 1)
                        sd = np.sqrt(variance, out=np.zeros_like(variance),
                                     where=variance >= 0.0)

                        pred_P = normalized_base_prediction * sd + mean
                        pred_SP = normalized_meta_prediction * sd + mean

                    # Gets faded errors for the related leaf predictors
                    fmae = self._get_predictors_faded_error(X[i])

                    # Selects, for each target, the best current performer
                    for j in range(self._n_targets):
                        b_pred = np.argmin([fmae['mean'][j],
                                            fmae['perceptron'][j],
                                            fmae['stacked_perceptron'][j]]
                                           )

                        if b_pred == 0:
                            # If all the expected errors are the same,
                            # use the standard perceptron
                            if fmae['mean'][j] == fmae['perceptron'][j] \
                                    == fmae['stacked_perceptron'][j]:
                                predictions[i, j] = pred_P[j]
                            # Otherwise, use the simplest approach
                            else:
                                predictions[i, j] = pred_M[j]
                        else:
                            if b_pred == 1:
                                # Use the stacked perceptron if its expected
                                # error is the same than the error for the
                                # standard perceptron
                                if fmae['perceptron'][j] == \
                                        fmae['stacked_perceptron'][j]:
                                    predictions[i, j] = pred_SP[j]
                                else:
                                    predictions[i, j] = pred_P[j]
                            else:
                                predictions[i, j] = pred_SP[j]

        return predictions

    def _attempt_to_split(self, node, parent, parent_idx: int):
        """Attempt to split a node.

        If there exists significant variance among the target space of the
        seen examples:

        1. Find split candidates and select the top 2.
        2. Compute the Hoeffding bound.
        3. If the difference between the merit ratio of the top 2 split
        candidates is smaller than 1 minus the Hoeffding bound:
           3.1 Replace the leaf node by a split node.
           3.2 Add a new leaf node on each branch of the new split node.
           3.3 Update tree's metrics

        Optional: Disable poor attribute. Depends on the tree's configuration.

        Parameters
        ----------
        node: ActiveLearningNode
            The node to evaluate.
        parent: SplitNode
            The node's parent.
        parent_idx: int
            Parent node's branch index.
        """
        split_criterion = IntraClusterVarianceReductionSplitCriterion()
        best_split_suggestions = node.\
            get_best_split_suggestions(split_criterion, self)

        best_split_suggestions.sort(key=attrgetter('merit'))
        should_split = False
        if len(best_split_suggestions) < 2:
            should_split = len(best_split_suggestions) > 0
        else:
            hoeffding_bound = self.compute_hoeffding_bound(
                split_criterion.get_range_of_merit(
                    node.get_observed_class_distribution()
                ), self.split_confidence, node.get_weight_seen())
            best_suggestion = best_split_suggestions[-1]
            second_best_suggestion = best_split_suggestions[-2]

            if best_suggestion.merit > 0.0 and \
                    (second_best_suggestion.merit / best_suggestion.merit <
                     1 - hoeffding_bound or hoeffding_bound <
                     self.tie_threshold):
                should_split = True
            if self.remove_poor_atts is not None and self.remove_poor_atts \
                    and not should_split:
                poor_atts = set()
                best_ratio = second_best_suggestion.merit \
                    / best_suggestion.merit

                # Add any poor attribute to set
                for i in range(len(best_split_suggestions)):
                    if best_split_suggestions[i].split_test is not None:
                        split_atts = best_split_suggestions[i].\
                            split_test.get_atts_test_depends_on()
                        if len(split_atts) == 1:
                            if best_split_suggestions[i].merit / \
                                    best_suggestion.merit < \
                                    best_ratio - 2 * hoeffding_bound:
                                poor_atts.add(int(split_atts[0]))

                for poor_att in poor_atts:
                    node.disable_attribute(poor_att)
        if should_split:
            split_decision = best_split_suggestions[-1]
            if split_decision.split_test is None:
                # Preprune - null wins
                self._deactivate_learning_node(node, parent, parent_idx)
            else:
                new_split = self.new_split_node(
                    split_decision.split_test,
                    node.get_observed_class_distribution()
                )
                for i in range(split_decision.num_splits()):
                    new_child = self._new_learning_node(
                        split_decision.resulting_class_distribution_from_split(i), node
                    )
                    new_split.set_child(i, new_child)
                self._active_leaf_node_cnt -= 1
                self._decision_node_cnt += 1
                self._active_leaf_node_cnt += split_decision.num_splits()
                if parent is None:
                    self._tree_root = new_split
                else:
                    parent.set_child(parent_idx, new_split)
            # Manage memory
            self.enforce_tracker_limit()
