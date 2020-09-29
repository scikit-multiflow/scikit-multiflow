from skmultiflow.data.source.data_source import DataSource
import threading


class DataframeDataSource(DataSource):
    """ DataframeDataSource class.
    Provides a DataSource implementation, reading from a dataframe.

    Parameters
    ----------
    dataframe: pd.DataFrame (Default=None)
        The features' columns and targets' columns or the feature columns
        only if they are passed separately.
    """
    def __init__(self, record_to_dictionary, observers, dataframe):
        super().__init__(record_to_dictionary, observers)
        self.dataframe = dataframe
        self.name = "DataframeDataSource" # TODO: can we md5 hash the content?
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
        for index, row in self.dataframe.iterrows():
            self.on_new_event(row)

    def get_info(self):
        return "DataframeDataSource; observers: {}".format([x.get_name() for x in self.observers])
