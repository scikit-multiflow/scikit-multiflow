from abc import ABCMeta, abstractmethod
from skmultiflow.core import BaseSKMObject
import threading


class DataSource(BaseSKMObject, metaclass=ABCMeta):
    """ DataSource class.

    This abstract class defines the minimum requirements of a data source.
    It provides an interface to access entries, streamed from certain source.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """
    _estimator_type = 'datasource'

    def __init__(self, record_to_dictionary, observers):
        self.name = None
        self.record_to_dictionary = record_to_dictionary
        self.observers = observers


    @abstractmethod
    def listen_for_events(self):
        raise NotImplementedError

    @abstractmethod
    def _prepare_for_use(self):
        """ Prepares the data source to be used
        """
        raise NotImplementedError

    def start_event_listener(self):
        self._prepare_for_use()
        t = threading.Thread(target=self.listen_for_events)
        t.start()

    def on_new_event(self, event):
        for observer in self.observers:
            observer.update(self.record_to_dictionary(event))

    def restart(self):
        """  Restart the stream. """
        if(self.is_restartable()):
            self.current_sample = None
            self._prepare_for_use()
        else:
            raise NotImplementedError

    def last_event(self):
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
