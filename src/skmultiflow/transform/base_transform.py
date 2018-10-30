from skmultiflow.core.base_object import BaseObject
from abc import ABCMeta, abstractmethod


class StreamTransform(BaseObject, metaclass=ABCMeta):
    """ BaseTransform
    
    Abstract class that explicits the constraints to all Transform objects 
    in this framework.
    
    This class should not be instantiated as it has no implementations and 
    will raise NotImplementedErrors.
    
    Raises
    ------
    NotImplementedError: This is an abstract class.
    
    """

    def __init__(self):
        super().__init__()

    def get_class_type(self):
        return 'transform'

    @abstractmethod
    def get_info(self):
        raise NotImplementedError

    @abstractmethod
    def transform(self, X):
        """ transform
        
        Transform the data in X and returns it
        
        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.
        
        Returns
        -------
        numpy.ndarray
            The transformed data
        
        """
        raise NotImplementedError

    @abstractmethod
    def partial_fit_transform(self, X, y=None):
        """ partial_fit_transform
        
        Partial fit and transform the Transformer object.
        
        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.
        
        y: Array-like
            An array-like with all the class labels from all samples in X.
        
        Returns
        -------
        StreamTransform
            The partially fitted model.
        
        """
        raise NotImplementedError

    @abstractmethod
    def partial_fit(self, X, y=None):
        """ partial_fit

        Partial fit the Transformer object.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            The feature's matrix.

        y: Array-like
            An array-like with all the class labels from all samples in X.

        Returns
        -------
        StreamTransform
            The partially fitted model.

        """
        raise NotImplementedError
