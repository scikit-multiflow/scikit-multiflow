import os
import threading
from skmultiflow.data.source.data_source import DataSource


class FileDataSource(DataSource):
    """ FileDataSource class.
    Provides a DataSource implementation, reading from a file.

    Examples
    --------
    >>> import time
    >>> from skmultiflow.data.source.file_data_source import FileDataSource
    >>> from skmultiflow.data.observer.buffer_data_event_observer import BufferDataEventObserver
    >>>
    >>>
    >>> def record_to_dictionary(record):
    >>>     record_array = record.strip().split(',')
    >>>     if(len(record_array)==5):
    >>>         return {'X': [record_array[0], record_array[1], record_array[2], record_array[3]], 'y': [record_array[4]]}
    >>>     return None
    >>>
    >>>
    >>> # Setup an event observer and a data source from file 'iris.data'
    >>> buffer_data_event_observer = BufferDataEventObserver()
    >>> data_source = FileDataSource(record_to_dictionary, [buffer_data_event_observer], 'iris.data')
    >>> data_source.listen_for_events()
    >>>
    >>> # Wait until there are at least two events in the buffer
    >>> while(len(buffer_data_event_observer.get_buffer())<2):
    >>>     time.sleep(0.100) # 100ms
    >>>
    >>> # Print first two events
    >>> first_event = buffer_data_event_observer.get_buffer()[0]
    >>> second_event = buffer_data_event_observer.get_buffer()[1]
    >>> print('First event: X: {}, y: {}'.format(first_event['X'], first_event['y']))
    >>> print('Second event: X: {}, y: {}'.format(second_event['X'], second_event['y']))

    """
    def __init__(self, record_to_dictionary, observers, filename):
        super().__init__(record_to_dictionary, observers)
        self.file_handler = None
        self.filename = filename
        self.name = "FileDataSource: {}".format(self.filename)
        self._prepare_for_use()

    def _prepare_for_use(self):
        """ Prepares the data source to be used
        """
        if (self.file_handler is not None):
            self.file_handler.close()
        self.file_handler = open(self.filename, "r")

    def listen_for_events(self):
        thread = threading.Thread(target=self.read_file, args=())
        thread.daemon = True
        thread.start()

    def read_file(self):
        event = self.file_handler.readline()
        while event is not '':
            self.on_new_event(event)
            event = self.file_handler.readline()
        self.file_handler.close()

    def get_info(self):
        return "FileDataSource: {}; observers: {}".format(os.path.basename(self.filename), [x.get_name() for x in self.observers])
