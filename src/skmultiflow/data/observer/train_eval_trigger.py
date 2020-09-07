from abc import ABCMeta, abstractmethod


class TrainEvalTrigger(metaclass=ABCMeta):
    """ TrainEvalTrigger class.

    This abstract class defines the minimum requirements of a trigger.
    It provides an interface to define criteria when data shall be fitted and evaluated.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """

    @abstractmethod
    def update(self, event):
        raise NotImplementedError

    @abstractmethod
    def shall_fit(self, event):
        raise NotImplementedError

    @abstractmethod
    def shall_predict(self, event):
        raise NotImplementedError
