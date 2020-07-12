from skmultiflow.trees.attribute_observer import NumericAttributeClassObserverGaussian
from skmultiflow.trees.attribute_observer import NominalAttributeClassObserver
from skmultiflow.trees.nodes import ActiveLearningNode
from skmultiflow.trees.nodes import InactiveLearningNode

from skmultiflow.bayes import do_naive_bayes_prediction


class LCActiveLearningNode(ActiveLearningNode):
    """ Active Learning node for the Label Combination Hoeffding Tree.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """
    def __init__(self, initial_stats):
        super().__init__(initial_stats)

    def learn_from_instance(self, X, y, weight, ht):

        if ht.leaf_prediction != ht._NAIVE_BAYES_ADAPTIVE:
            y = ''.join(str(e) for e in y)
            y = int(y, 2)

        try:
            self._stats[y] += weight
        except KeyError:
            self._stats[y] = weight

        for i in range(len(X)):
            try:
                obs = self._attribute_observers[i]
            except KeyError:
                if ht.nominal_attributes is not None and i in ht.nominal_attributes:
                    obs = NominalAttributeClassObserver()
                else:
                    obs = NumericAttributeClassObserverGaussian()
                self._attribute_observers[i] = obs
            obs.update(X[i], int(y), weight)


class LCInactiveLearningNode(InactiveLearningNode):
    """ Inactive Learning node for the Label Combination Hoeffding Tree.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """
    def __init__(self, initial_stats=None):
        """ LCInactiveLearningNode class constructor. """
        super().__init__(initial_stats)

    def learn_from_instance(self, X, y, weight, ht):

        i = ''.join(str(e) for e in y)
        i = int(i, 2)
        try:
            self._stats[i] += weight
        except KeyError:
            self._stats[i] = weight


class LCLearningNodeNB(LCActiveLearningNode):
    """ Learning node for the Label Combination Hoeffding Tree that uses Naive
    Bayes models.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """
    def __init__(self, initial_stats):
        """ LCLearningNodeNB class constructor. """
        super().__init__(initial_stats)

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
        if self.total_weight >= ht.nb_threshold:
            return do_naive_bayes_prediction(
                X, self._stats, self._attribute_observers
            )
        else:
            return super().get_class_votes(X, ht)

    def disable_attribute(self, att_index):
        """ Disable an attribute observer.

        Disabled in Nodes using Naive Bayes, since poor attributes are used in
        Naive Bayes calculation.

        Parameters
        ----------
        att_index: int
            Attribute index.

        """
        pass


class LCLearningNodeNBA(LCLearningNodeNB):
    """ Learning node for the Label Combination Hoeffding Tree that uses
    Adaptive Naive Bayes models.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """
    def __init__(self, initial_stats):
        """LCLearningNodeNBA class constructor. """
        super().__init__(initial_stats)
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

        y = ''.join(str(e) for e in y)
        y = int(y, 2)

        if self._stats == {}:
            # All target_values equal, default to class 0
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
