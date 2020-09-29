from skmultiflow.data.source.data_source import DataSource
import threading


class ArrayDataSource(DataSource):
    """ ArrayDataSource class.
    Provides a DataSource implementation, reading from an array.

    Parameters
    ----------
    array: np.array (Default=None)
        The features' columns and targets' columns or the feature columns
        only if they are passed separately.
    """
    def __init__(self, record_to_dictionary, observers, array):
        super().__init__(record_to_dictionary, observers)
        self.array = array
        self.name = "ArrayDataSource" # TODO: can we md5 hash the content?
        self._prepare_for_use()

    def _prepare_for_use(self):
        """ Prepares the data source to be used
        """
        pass

    def listen_for_events(self):
        thread = threading.Thread(target=self.read_content, args=())
        thread.daemon = True
        thread.start()

    def read_content(self):
        for j in range(0, len(self.array)):
            self.on_new_event(self.array[j])

    def get_info(self):
        return "ArrayDataSource; observers: {}".format([x.get_name() for x in self.observers])
