from skmultiflow.trees.nodes import Node


class LearningNode(Node):
    """ Base class for Learning Nodes in a Hoeffding Tree.

    Parameters
    ----------
    initial_stats: dict (class_value, weight) or None
        Initial class observations

    """

    def __init__(self, initial_stats=None):
        """ LearningNode class constructor. """
        super().__init__(initial_stats)

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
        pass
