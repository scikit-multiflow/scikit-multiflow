from abc import ABCMeta, abstractmethod


class MetricsReporter(metaclass=ABCMeta):
    """ MetricsReporter class.

    This abstract class defines the minimum requirements of a data source.
    It provides an interface to access entries, streamed from certain source.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """

    @abstractmethod
    def report(self, measurements):
        raise NotImplementedError
