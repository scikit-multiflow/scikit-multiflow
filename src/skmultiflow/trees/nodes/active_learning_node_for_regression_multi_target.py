from skmultiflow.trees.nodes import ActiveLearningNodeForRegression
from skmultiflow.trees.attribute_observer import NumericAttributeRegressionObserverMultiTarget
from skmultiflow.trees.attribute_observer import NominalAttributeRegressionObserver


class ActiveLearningNodeForRegressionMultiTarget(ActiveLearningNodeForRegression):

    def __init__(self, initial_class_observations):
        """ ActiveLearningNodeForRegressionMultiTarget class constructor. """
        super().__init__(initial_class_observations)

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
            self._observed_class_distribution[0] += weight
            self._observed_class_distribution[1] += y * weight
            self._observed_class_distribution[2] += y * y * weight
        except KeyError:
            self._observed_class_distribution[0] = weight
            self._observed_class_distribution[1] = y * weight
            self._observed_class_distribution[2] = y * y * weight

        for i, x in enumerate(X.tolist()):
            try:
                obs = self._attribute_observers[i]
            except KeyError:
                if ht.nominal_attributes is not None and i in ht.nominal_attributes:
                    obs = NominalAttributeRegressionObserver()
                else:
                    obs = NumericAttributeRegressionObserverMultiTarget()
                self._attribute_observers[i] = obs
            obs.observe_attribute_class(x, y, weight)
