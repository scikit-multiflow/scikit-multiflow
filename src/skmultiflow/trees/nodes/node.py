import textwrap
from abc import ABCMeta
from skmultiflow.trees.nodes import FoundNode


class Node(metaclass=ABCMeta):
    """ Base class for nodes in a Hoeffding Tree.

    Parameters
    ----------
    stats: dict (class_value, weight) or None
        Class observations.

    """

    def __init__(self, stats=None):
        """ Node class constructor. """
        if stats is None:
            stats = {}  # Dictionary (class_value, weight)
        self._stats = stats

    @staticmethod
    def is_leaf():
        """ Determine if the node is a leaf.

        Returns
        -------
        True if leaf, False otherwise

        """
        return True

    def filter_instance_to_leaf(self, X, parent, parent_branch):
        """ Traverse down the tree to locate the corresponding leaf for an instance.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
           Data instances.
        parent: skmultiflow.trees.nodes.Node or None
            Parent node.
        parent_branch: Int
            Parent branch index

        Returns
        -------
        FoundNode
            The corresponding leaf.

        """
        return FoundNode(self, parent, parent_branch)

    def get_stats(self):
        """ Get the current observed class distribution at the node.

        Returns
        -------
        dict (class_value, weight)
            Class distribution at the node.

        """
        return self._stats

    def set_stats(self, observed_class_distribution):
        """ Set the observed class distribution at the node.

        Parameters
        -------
        dict (class_value, weight)
            Class distribution at the node.

        """
        self._stats = observed_class_distribution

    def get_class_votes(self, X, ht):
        """ Get the votes per class for a given instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
           Data instances.
        ht: HoeffdingTreeClassifier
            The Hoeffding Tree.

        Returns
        -------
        dict (class_value, weight)
            Class votes for the given instance.

        """
        return self._stats

    def observed_class_distribution_is_pure(self):
        """ Check if observed class distribution is pure, i.e. if all samples
        belong to the same class.

        Returns
        -------
        boolean
            True if observed number of classes is less than 2, False otherwise.

        """
        count = 0
        for _, weight in self._stats.items():
            if weight != 0:
                count += 1
                if count == 2:  # No need to count beyond this point
                    break
        return count < 2

    def subtree_depth(self):
        """ Calculate the depth of the subtree from this node.

        Returns
        -------
        int
            Subtree depth, 0 if the node is a leaf.

        """
        return 0

    def calculate_promise(self):
        """ Calculate node's promise.

        Returns
        -------
        int
            A small value indicates that the node has seen more samples of a
            given class than the other classes.

        """
        total_seen = sum(self._stats.values())
        if total_seen > 0:
            return total_seen - max(self._stats.values())
        else:
            return 0

    def describe_subtree(self, ht, buffer, indent=0):
        """ Walk the tree and write its structure to a buffer string.

        Parameters
        ----------
        ht: HoeffdingTreeClassifier
            The tree to describe.
        buffer: string
            The string buffer where the tree's structure will be stored
        indent: int
            Indentation level (number of white spaces for current node.)

        """
        buffer[0] += textwrap.indent('Leaf = ', ' ' * indent)

        if ht._estimator_type == 'classifier':
            class_val = max(
                self._stats,
                key=self._stats.get
            )
            buffer[0] += 'Class {} | {}\n'.format(
                class_val, self._stats
            )
        else:
            text = '{'
            for i, (k, v) in enumerate(self._stats.items()):
                # Multi-target regression case
                if hasattr(v, 'shape') and len(v.shape) > 0:
                    text += '{}: ['.format(k)
                    text += ', '.join(['{:.4f}'.format(e) for e in v.tolist()])
                    text += ']'
                else:  # Single-target regression
                    text += '{}: {:.4f}'.format(k, v)
                text += ', ' if i < len(self._stats) - 1 else ''
            text += '}'
            buffer[0] += 'Statistics {}\n'.format(text)  # Regression problems

    # TODO
    def get_description(self):
        pass
