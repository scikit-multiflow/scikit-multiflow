from abc import ABCMeta, abstractmethod
from skmultiflow.core import BaseSKMObject
from kafka import KafkaConsumer
import warnings

class KafkaDataSource(DataSource):
    """ KafkaDataSource class.

    Provides a DataSource implementation, reading from Kafka consumer.

    """
    _estimator_type = 'datasource'

    def __init__(self, kafka_topic, kafka_group_id, bootstrap_servers, record_to_dictionary):
        self.topic = kafka_topic
        self.group_id = kafka_group_id
        self.bootstrap_servers = bootstrap_servers
        self.name = "KafkaDataSource: {}{}[at]{}".format(self.topic, self.group_id, self.bootstrap_servers)
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
        entry = self.consumer.get_message(block=False, timeout=self.IterTimeout, get_partition_info=True)
        if(entry is None):
            self.current_sample = None
        else:
            self.current_sample = self.record_to_dictionary(entry)

        return self.current_sample

    def _prepare_for_use(self):
        """ Prepares the data source to be used
        """
        if(self.group_id == None):
            self.consumer = KafkaConsumer(self.topic, self.group_id, bootstrap_servers=self.bootstrap_servers,
                                          auto_offset_reset='earliest', enable_auto_commit=True,
                                          auto_commit_interval_ms=1000)
        else:
            self.consumer = KafkaConsumer(self.topic, bootstrap_servers=self.bootstrap_servers,
                                          auto_offset_reset='earliest', enable_auto_commit=True,
                                          auto_commit_interval_ms=1000)
