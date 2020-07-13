import numpy as np

from skmultiflow.trees.nodes import ActiveLeafClass, LearningNodeMC, LearningNodeNB, \
    LearningNodeNBA

from skmultiflow.utils import get_dimensions


class RandomActiveLeafClass(ActiveLeafClass):
    """ Random Active Leaf

    A Random Active Leaf (used in ARF implementations) just changes the way how the nodes update
    the attribute observers (by using subsets of features).
    """
    def update_attribute_observers(self, X, y, weight, tree):
        if self.list_attributes.size == 0:
            self.list_attributes = self._sample_features(get_dimensions(X)[1])

        for idx in self.list_attributes:
            try:
                obs = self.attribute_observers[idx]
            except KeyError:
                if tree.nominal_attributes is not None and idx in tree.nominal_attributes:
                    obs = self.get_nominal_attribute_observer()
                else:
                    obs = self.get_numeric_attribute_observer()
                self.attribute_observers[idx] = obs
            obs.update(X[idx], y, weight)

    def _sample_features(self, n_features):
        return self.random_state.choice(
            n_features, size=self.max_features, replace=False
        )


class RandomActiveLearningNodeMC(LearningNodeMC, RandomActiveLeafClass):
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


class RandomActiveLearningNodeNB(LearningNodeNB, RandomActiveLeafClass):
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
        super().__init__(initial_stats)
        self.max_features = max_features
        self.list_attributes = np.array([])
        self.random_state = random_state


class RandomActiveLearningNodeNBA(LearningNodeNBA, RandomActiveLeafClass):
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
        super().__init__(initial_stats)
        self.max_features = max_features
        self.list_attributes = np.array([])
        self.random_state = random_state
