from skmultiflow.trees.attribute_split_suggestion import AttributeSplitSuggestion
from skmultiflow.trees.attribute_observer import AttributeClassObserverNull
from skmultiflow.trees.attribute_observer import NominalAttributeClassObserver
from skmultiflow.trees.attribute_observer import NumericAttributeClassObserverGaussian
from skmultiflow.trees.nodes import LearningNode
from skmultiflow.bayes import do_naive_bayes_prediction


class ActiveLearningNode(LearningNode):
    """ Learning node that supports growth.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_stats):
        """ ActiveLearningNode class constructor. """
        super().__init__(initial_stats)
        self._weight_seen_at_last_split_evaluation = self.get_weight_seen()
        self._attribute_observers = {}

    def learn_from_instance(self, X, y, weight, ht):
        """ Update the node with the provided instance.

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
                sorted(self._stats.items()))

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

    def get_weight_seen(self):
        """ Calculate the total weight seen by the node.

        Returns
        -------
        float
            Total weight seen.

        """
        return sum(self._stats.values())

    def get_weight_seen_at_last_split_evaluation(self):
        """ Retrieve the weight seen at last split evaluation.

        Returns
        -------
        float
            Weight seen at last split evaluation.

        """
        return self._weight_seen_at_last_split_evaluation

    def set_weight_seen_at_last_split_evaluation(self, weight):
        """ Set the weight seen at last split evaluation.

        Parameters
        ----------
        weight: float
            Weight seen at last split evaluation.

        """
        self._weight_seen_at_last_split_evaluation = weight

    def get_best_split_suggestions(self, criterion, ht):
        """ Find possible split candidates.

        Parameters
        ----------
        criterion: SplitCriterion
            The splitting criterion to be used.
        ht: HoeffdingTreeClassifier
            Hoeffding Tree.

        Returns
        -------
        list
            Split candidates.

        """
        best_suggestions = []
        pre_split_dist = self._stats
        if not ht.no_preprune:
            # Add null split as an option
            null_split = AttributeSplitSuggestion(
                None, [{}], criterion.get_merit_of_split(pre_split_dist, [pre_split_dist])
            )
            best_suggestions.append(null_split)
        for i, obs in self._attribute_observers.items():
            best_suggestion = obs.get_best_evaluated_split_suggestion(
                criterion, pre_split_dist, i, ht.binary_split
            )
            if best_suggestion is not None:
                best_suggestions.append(best_suggestion)
        return best_suggestions

    def disable_attribute(self, att_idx):
        """ Disable an attribute observer.

        Parameters
        ----------
        att_idx: int
            Attribute index.

        """
        if att_idx in self._attribute_observers:
            self._attribute_observers[att_idx] = AttributeClassObserverNull()

    def get_attribute_observers(self):
        """ Get attribute observers at this node.

        Returns
        -------
        dict (attribute id, attribute observer object)
            Attribute observers of this node.

        """
        return self._attribute_observers


class InactiveLearningNode(LearningNode):
    """ Inactive learning node that does not grow.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_stats=None):
        """ InactiveLearningNode class constructor. """
        super().__init__(initial_stats)

    def learn_from_instance(self, X, y, weight, ht):
        """ Update the node with the provided instance.

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
                sorted(self._stats.items()))


class ActiveLearningNodeNB(ActiveLearningNode):
    """ Learning node that uses Naive Bayes models.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_stats):
        """ LearningNodeNB class constructor. """
        super().__init__(initial_stats)

    def get_class_votes(self, X, ht):
        """ Get the votes per class for a given instance.

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


class ActiveLearningNodeNBAdaptive(ActiveLearningNodeNB):
    """ Learning node that uses Adaptive Naive Bayes models.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_stats):
        """ LearningNodeNBAdaptive class constructor. """
        super().__init__(initial_stats)
        self._mc_correct_weight = 0.0
        self._nb_correct_weight = 0.0

    def learn_from_instance(self, X, y, weight, ht):
        """ Update the node with the provided instance.

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
        """ Get the votes per class for a given instance.

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
