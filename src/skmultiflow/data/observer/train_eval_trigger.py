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
        """
        This method aims to store the event, so that can be used to fit or predict
        :param event:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def shall_fit(self):
        raise NotImplementedError

    @abstractmethod
    def shall_predict(self):
        raise NotImplementedError
