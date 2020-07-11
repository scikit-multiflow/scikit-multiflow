import numpy as np

from skmultiflow.trees.nodes import ActiveLearningNode
from skmultiflow.trees.attribute_observer import NominalAttributeClassObserver
from skmultiflow.trees.attribute_observer import NumericAttributeClassObserverGaussian

from skmultiflow.bayes import do_naive_bayes_prediction
from skmultiflow.utils import get_dimensions


class RandomActiveLearningNode(ActiveLearningNode):
    """ARF learning node class.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations.

    max_features: int
        Number of attributes per subset for each node split.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_stats, max_features, random_state=None):
        """ RandomLearningNodeClassification class constructor. """
        super().__init__(initial_stats)

        self.max_features = max_features
        self.list_attributes = np.array([])
        self.random_state = random_state

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
        ht: HoeffdingTreeClassifier
            Hoeffding Tree to update.
        """
        try:
            self._stats[y] += weight
        except KeyError:
            self._stats[y] = weight
            self._stats = dict(
                sorted(self._stats.items())
            )

        if self.list_attributes.size == 0:
            self.list_attributes = self._sample_features(get_dimensions(X)[1])

        for i in self.list_attributes:
            try:
                obs = self._attribute_observers[i]
            except KeyError:
                if ht.nominal_attributes is not None and i in ht.nominal_attributes:
                    obs = NominalAttributeClassObserver()
                else:
                    obs = NumericAttributeClassObserverGaussian()
                self._attribute_observers[i] = obs
            obs.update(X[i], int(y), weight)

    def _sample_features(self, n_features):
        return self.random_state.choice(
            n_features, size=self.max_features, replace=False
        )


class RandomActiveLearningNodeNB(RandomActiveLearningNode):
    """ARF Naive Bayes learning node class.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations.

    max_features: int
        Number of attributes per subset for each node split.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """

    def __init__(self, initial_stats, max_features, random_state):
        """ LearningNodeNB class constructor. """
        super().__init__(initial_stats, max_features, random_state)

    def get_class_votes(self, X, ht):
        """Get the votes per class for a given instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.
        ht: HoeffdingTreeClassifier
            Hoeffding Tree.

        Returns
        -------
        dict (class_value, weight)
            Class votes for the given instance.

        """
        if self.get_weight_seen() >= ht.nb_threshold:
            return do_naive_bayes_prediction(
                X, self._stats, self._attribute_observers
            )
        else:
            return super().get_class_votes(X, ht)


class RandomActiveLearningNodeNBAdaptive(RandomActiveLearningNodeNB):
    """Naive Bayes Adaptive learning node class.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations.

    max_features: int
        Number of attributes per subset for each node split.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    """
    def __init__(self, initial_stats, max_features, random_state):
        """LearningNodeNBAdaptive class constructor. """
        super().__init__(initial_stats, max_features, random_state)
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
        ht: HoeffdingTreeClassifier
            The Hoeffding Tree to update.

        """
        if self._stats == {}:
            # All classes equal, default to class 0
            if 0 == y:
                self._mc_correct_weight += weight
        elif max(self._stats,
                 key=self._stats.get) == y:
            self._mc_correct_weight += weight
        nb_prediction = do_naive_bayes_prediction(
            X, self._stats, self._attribute_observers
        )
        if max(nb_prediction, key=nb_prediction.get) == y:
            self._nb_correct_weight += weight
        super().learn_from_instance(X, y, weight, ht)

    def get_class_votes(self, X, ht):
        """Get the votes per class for a given instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.
        ht: HoeffdingTreeClassifier
            Hoeffding Tree.

        Returns
        -------
        dict (class_value, weight)
            Class votes for the given instance.

        """
        if self._mc_correct_weight > self._nb_correct_weight:
            return self._stats
        return do_naive_bayes_prediction(
            X, self._stats, self._attribute_observers
        )
