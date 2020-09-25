from skmultiflow.data.source.data_source import DataSource
import threading


class KafkaDataSource(DataSource):
    """ KafkaDataSource class.
    Provides a DataSource implementation, reading from Kafka consumer.
    """

    def __init__(self, record_to_dictionary, observers, kafka_consumer):
        super().__init__(record_to_dictionary, observers)
        consumer_properties = kafka_consumer.__dict__['config']
        self.name = "KafkaDataSource: bootstrap_servers: {}; group_id: {}".format(consumer_properties['bootstrap_servers'], consumer_properties['group_id'])
        self.kafka_consumer = kafka_consumer
        self._prepare_for_use()

    def _prepare_for_use(self):
        """ Prepares the data source to be used
        """
        return None

    def listen_for_events(self):
        thread = threading.Thread(target=self.consume_kafka_messages, args=())
        thread.daemon = True
        thread.start()

    def consume_kafka_messages(self):
        for message in self.kafka_consumer:
            self.on_new_event(message.value)

    def get_info(self):
        return self.name
