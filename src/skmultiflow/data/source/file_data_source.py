from skmultiflow.data.source import DataSource


class FileDataSource(DataSource):
    """ FileDataSource class.
    Provides a DataSource implementation, reading from a file.
    """

    def __init__(self, record_to_dictionary, observers, filename):
        self.record_to_dictionary = record_to_dictionary
        self.observers = observers
        self.filename = filename
        self.name = "FileDataSource: {}".format(self.filename)
        self._prepare_for_use()

    def _prepare_for_use(self):
        """ Prepares the data source to be used
        """
        if(self.file_handler is not None):
            self.file_handler.close()
        self.file_handler = open(self.filename, "r")


    def listen_for_events(self):
        event = self.file_handler.readline()
        while event is not None:
            self.on_new_event(event)
            event = self.file_handler.readline()

        self.file_handler.close()