from abc import ABCMeta, abstractmethod
from skmultiflow.core import BaseSKMObject
import warnings

class FileDataSource(DataSource):
    """ FileDataSource class.

    Provides a DataSource implementation, reading from a file.

    """
    _estimator_type = 'datasource'

    def __init__(self, filename, record_to_dictionary):
        self.filename = filename
        self.name = "FileDataSource: {}".format(self.filename)
        self.record_to_dictionary = record_to_dictionary
        self._prepare_for_use()

    def next_sample(self):
        """ Returns a dictionary with data of next sample in the stream.

        Parameters
        ----------

        Returns
        -------
        tuple - sample

        """
        entry = self.file_handler.readline()
        if(entry == None):
            self.file_handler.close()
            self.current_sample = None
        else:
            self.current_sample = self.record_to_dictionary(entry)
        return self.current_sample

    def _prepare_for_use(self):
        """ Prepares the data source to be used
        """
        if(self.file_handler is not None):
            self.file_handler.close()
        self.file_handler = open(self.filename, "r")
        self.current_sample = None
