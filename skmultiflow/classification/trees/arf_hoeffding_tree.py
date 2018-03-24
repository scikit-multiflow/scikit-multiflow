__author__ = 'Anderson Carlos Ferreira da Silva'

from random import sample
from skmultiflow.classification.trees.hoeffding_tree import *


class ARFHoeffdingTree(HoeffdingTree):
    """ Adaptive Random Forest Hoeffding Tree.

    Parameters
    __________
    max_byte_size: int (default=33554432)
        Maximum memory consumed by the tree.
    memory_estimate_period: int (default=1000000)
        Number of instances between memory consumption checks.
    grace_period: int (default=200)
        Number of instances a leaf should observe between split attempts.
    split_criterion: string (default='info_gain')
        | Split criterion to use.
        | 'gini' - Gini
        | 'info_gain' - Information Gain
    split_confidence: float (default=0.0000001)
        Allowed error in split decision, a value closer to 0 takes longer to decide.
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
    leaf_prediction: string (default='nba')
        | Prediction mechanism used at leafs.
        | 'mc' - Majority Class
        | 'nb' - Naive Bayes
        | 'nba' - Naive Bayes Adaptive
    nb_threshold: int (default=0)
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.
    max_features: int (default=2)
            Max number of attributes for each node split.

    Notes
    _____
    This is the base model for the Adaptive Random Forest ensemble learner
    (See skmultiflow.classification.meta.adaptive_random_forests).
    This Hoeffding Tree includes a subspace size parameter, which defines the number of randomly selected features to
    be considered at each split.

    """
    class RandomLearningNode(HoeffdingTree.ActiveLearningNode):
        """Random learning node class.

        Parameters
        ----------
        initial_class_observations: dict (class_value, weight) or None
            Initial class observations
        subspace_size: int
            Number of attributes per subset for each node split.

        """
        def __init__(self,
                     initial_class_observations,
                     subspace_size):
            """ RandomLearningNode class constructor. """
            super().__init__(initial_class_observations)

            self.subspace_size = subspace_size
            self._attribute_observers = {}
            self.list_attributes = []

        def learn_from_instance(self, X, y, weight, ht):
            """Update the node with the provided instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
            y: int
                Instance class.
            weight: float
                Instance weight.
            ht: HoeffdingTree
                Hoeffding Tree to update.

            """
            try:
                self._observed_class_distribution[y] += weight
            except KeyError:
                self._observed_class_distribution[y] = weight
            if not self.list_attributes:
                self.list_attributes = sample(range(get_dimensions(X)[1]), self.subspace_size)  # TODO check attr-1

            for i in self.list_attributes:
                try:
                    obs = self._attribute_observers[i]
                except KeyError:
                    if i in ht.nominal_attributes:
                        obs = NominalAttributeClassObserver()
                    else:
                        obs = GaussianNumericAttributeClassObserver()
                    self._attribute_observers[i] = obs
                obs.observe_attribute_class(X[i], int(y), weight)

    class LearningNodeNB(RandomLearningNode):
        """Naive Bayes learning node class.

        Parameters
        ----------
        initial_class_observations: dict (class_value, weight) or None
            Initial class observations
        subspace_size: int
            Number of attributes per subset for each node split.

        """
        def __init__(self, initial_class_observations, subspace_size):
            """ LearningNodeNB class constructor. """
            super().__init__(initial_class_observations, subspace_size)

        def get_class_votes(self, X, ht):
            """Get the votes per class for a given instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes.
            ht: HoeffdingTree
                Hoeffding Tree.

            Returns
            -------
            dict (class_value, weight)
                Class votes for the given instance.

            """
            if self.get_weight_seen() >= ht.nb_threshold:
                return do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            else:
                return super().get_class_votes(X, ht)

    class LearningNodeNBAdaptive(LearningNodeNB):
        """Naive Bayes Adaptive learning node class.

        Parameters
        ----------
        initial_class_observations: dict (class_value, weight) or None
            Initial class observations
        subspace_size: int
            Number of attributes per subset for each node split.

        """
        def __init__(self, initial_class_observations, subspace_size):
            """LearningNodeNBAdaptive class constructor. """
            super().__init__(initial_class_observations, subspace_size)
            self._mc_correct_weight = 0.0
            self._nb_correct_weight = 0.0

        def learn_from_instance(self, X, y, weight, ht):
            """Update the node with the provided instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes for updating the node.
            y: int
                Instance class.
            weight: float
                The instance's weight.
            ht: HoeffdingTree
                The Hoeffding Tree to update.

            """
            if self._observed_class_distribution == {}:
                # All classes equal, default to class 0
                if 0 == y:
                    self._mc_correct_weight += weight
            elif max(self._observed_class_distribution, key=self._observed_class_distribution.get) == y:
                self._mc_correct_weight += weight
            nb_prediction = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            if max(nb_prediction, key=nb_prediction.get) == y:
                self._nb_correct_weight += weight
            super().learn_from_instance(X, y, weight, ht)

        def get_class_votes(self, X, ht):
            """Get the votes per class for a given instance.

            Parameters
            ----------
            X: numpy.ndarray of length equal to the number of features.
                Instance attributes.
            ht: HoeffdingTree
                Hoeffding Tree.

            Returns
            -------
            dict (class_value, weight)
                Class votes for the given instance.

            """
            if self._mc_correct_weight > self._nb_correct_weight:
                return self._observed_class_distribution
            return do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)

    def __init__(self,
                 max_byte_size=33554432,
                 memory_estimate_period=2000000,
                 grace_period=50,
                 split_criterion='info_gain',
                 split_confidence=0.01,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_atts=False,
                 no_preprune=False,
                 leaf_prediction='nba',
                 nb_threshold=0,
                 nominal_attributes=None,
                 max_features=2):
        """ADFHoeffdingTree class constructor."""
        # TODO Add HT parameters to ARF Hoeffding Tree constructor signature
        super().__init__(max_byte_size,
                         memory_estimate_period,
                         grace_period,
                         split_criterion,
                         split_confidence,
                         tie_threshold,
                         binary_split,
                         stop_mem_management,
                         remove_poor_atts,
                         no_preprune,
                         leaf_prediction,
                         nb_threshold,
                         nominal_attributes)
        self.max_features = max_features
        self.remove_poor_attributes_option = None

    def _new_learning_node(self, initial_class_observations=None):
        """Create a new learning node. The type of learning node depends on the tree configuration."""
        if initial_class_observations is None:
            initial_class_observations = {}
        # MAJORITY CLASS
        if self._leaf_prediction == MAJORITY_CLASS:
            return self.RandomLearningNode(initial_class_observations, self.max_features)
        # NAIVE BAYES
        elif self._leaf_prediction == NAIVE_BAYES:
            return self.LearningNodeNB(initial_class_observations, self.max_features)
        # NAIVE BAYES ADAPTIVE
        else:
            return self.LearningNodeNBAdaptive(initial_class_observations, self.max_features)

    @staticmethod
    def is_randomizable():
        return True

    def new_instance(self):
        return ARFHoeffdingTree(nominal_attributes=self.nominal_attributes, max_features=self.max_features)
        # TODO Pass all HT parameters once they are available at the ARFHT class level
