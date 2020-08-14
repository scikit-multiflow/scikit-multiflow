from abc import ABCMeta, abstractmethod
from skmultiflow.core import BaseSKMObject
import warnings
import threading

#OK
class EventObsever(BaseSKMObject, metaclass=ABCMeta):
    """ EventObserver class.

    This abstract class defines the minimum requirements of a data source.
    It provides an interface to access entries, streamed from certain source.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """
    _estimator_type = 'eventobserver'

    @abstractmethod
    def update(self, event):
        raise NotImplementedError
