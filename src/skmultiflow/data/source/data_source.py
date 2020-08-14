from abc import ABCMeta, abstractmethod
from skmultiflow.core import BaseSKMObject
import warnings

#OK
class DataSource(BaseSKMObject, metaclass=ABCMeta):
    """ DataSource class.

    This abstract class defines the minimum requirements of a data source.
    It provides an interface to access entries, streamed from certain source.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """
    _estimator_type = 'datasource'

    def __init__(self, record_to_dictionary):
        self.name = None
        self.record_to_dictionary = record_to_dictionary

    @abstractmethod
    def next_sample(self):
        """ Returns a dictionary with data of next sample in the stream.

        Parameters
        ----------

        Returns
        -------
        tuple - sample

        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_for_use(self):
        """ Prepares the data source to be used
        """
        raise NotImplementedError

    def restart(self):
        """  Restart the stream. """
        if(self.is_restartable()):
            self.current_sample = None
            self._prepare_for_use()
        else:
            raise NotImplementedError

    def last_sample(self):
        """ Retrieves last sample returned from source.

        Returns
        -------
        dictionary - last sample

        """
        return self.current_sample

    def is_restartable(self):
        """
        Determine if the source can restart sending data.

        Returns
        -------
        Bool
            True if stream is restartable.

        """
        return False

    def get_source_info(self):
        """ Retrieves source description.

        The default format is: {'name': '', 'tuple_params':[list of param keys]}.

        Returns
        -------
        string
            Source data information

        """
        return self.name
