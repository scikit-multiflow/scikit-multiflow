from skmultiflow.data.source import DataSource
from kafka import KafkaConsumer

class KafkaDataSource(DataSource):
    """ KafkaDataSource class.
    Provides a DataSource implementation, reading from Kafka consumer.
    """

    def __init__(self, record_to_dictionary, observers, bootstrap_servers, kafka_topic, kafka_group_id=None, ):
        self.record_to_dictionary = record_to_dictionary
        self.observers = observers
        self.bootstrap_servers = bootstrap_servers
        self.topic = kafka_topic
        self.group_id = kafka_group_id
        self.name = "KafkaDataSource: {}{}[at]{}".format(self.topic, self.group_id, self.bootstrap_servers)

        self._prepare_for_use()

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

    def listen_for_events(self):
        for message in self.consumer:
            self.on_new_event(message.value)