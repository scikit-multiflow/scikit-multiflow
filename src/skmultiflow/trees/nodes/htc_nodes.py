from skmultiflow.trees.nodes import LearningNodeMixin, ActiveLeafMixin, InactiveLeafMixin
from skmultiflow.bayes import do_naive_bayes_prediction


class LearningNodeMC(LearningNodeMixin):
    def update_stats(self, y, weight):
        try:
            self.stats[y] += weight
        except KeyError:
            self.stats[y] = weight
            self.stats = dict(sorted(self.stats.items()))

    def learn_one(self, X, y, *, weight=1.0, tree=None):
        # Enforce y is an integer
        super().learn_one(X, int(y), weight=weight, tree=tree)

    def predict_one(self, X, *, tree=None):
        return self.stats

    @property
    def total_weight(self):
        """ Calculate the total weight seen by the node.

        Returns
        -------
        float
            Total weight seen.

        """
        return sum(self._stats.values())


class LearningNodeNB(LearningNodeMC):
    def predict_one(self, X, *, tree=None):
        if self.total_weight >= tree.nb_threshold:
            return do_naive_bayes_prediction(
                X, self.stats, self.attribute_observers
            )
        else:
            return self.stats


class LearningNodeNBA(LearningNodeMC):
    def learn_one(self, X, y, *, weight=1.0, tree=None):
        """ Update the node with the provided instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes for updating the node.
        y: int
            Instance class.
        weight: float
            The instance's weight.
        tree: HoeffdingTreeClassifier
            The Hoeffding Tree to update.

        """
        if self.stats == {}:
            # All classes equal, default to class 0
            if y == 0:
                self._mc_correct_weight += weight
        elif max(self.stats, key=self.stats.get) == y:
            self._mc_correct_weight += weight
        nb_prediction = do_naive_bayes_prediction(
            X, self.stats, self.attribute_observers
        )
        if max(nb_prediction, key=nb_prediction.get) == y:
            self._nb_correct_weight += weight

        super().learn_from_instance(X, y, weight, tree)

    def predict_one(self, X, *, tree=None):
        """ Get the votes per class for a given instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.
        tree: HoeffdingTreeClassifier
            Hoeffding Tree.

        Returns
        -------
        dict (class_value, weight)
            Class votes for the given instance.

        """
        if self._mc_correct_weight > self._nb_correct_weight:
            return self.stats
        return do_naive_bayes_prediction(
            X, self.stats, self.attribute_observers
        )


class ActiveLearningNodeMC(LearningNodeMC, ActiveLeafMixin):
    """ Learning node that supports growth.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations
    """

    def __init__(self, initial_stats):
        """ ActiveLearningNode class constructor. """
        super().__init__(initial_stats)
        self.last_split_attempt_at = self.total_weight


class InactiveLearningNodeMC(LearningNodeMC, InactiveLeafMixin):
    """ Inactive learning node that does not grow.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations
    """

    def __init__(self, initial_stats=None):
        """ InactiveLearningNode class constructor. """
        super().__init__(initial_stats)


class ActiveLearningNodeNB(LearningNodeNB, ActiveLeafMixin):
    """ Learning node that uses Naive Bayes models.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations
    """

    def __init__(self, initial_stats):
        """ LearningNodeNB class constructor. """
        super().__init__(initial_stats)
        self.last_split_attempt_at = self.total_weight

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


class ActiveLearningNodeNBA(LearningNodeNBA, ActiveLeafMixin):
    """ Learning node that uses Adaptive Naive Bayes models.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations
    """

    def __init__(self, initial_stats):
        """ LearningNodeNBAdaptive class constructor. """
        super().__init__(initial_stats)
        self.last_split_attempt_at = self.total_weight
        self._mc_correct_weight = 0.0
        self._nb_correct_weight = 0.0

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
