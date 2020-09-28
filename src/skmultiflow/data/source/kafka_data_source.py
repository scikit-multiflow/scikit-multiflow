from skmultiflow.data.source.data_source import DataSource
import threading


class KafkaDataSource(DataSource):
    """ KafkaDataSource class.
    Provides a DataSource implementation, reading from Kafka consumer.

    Examples
    --------
    >>> from kafka import SimpleProducer, KafkaClient, KafkaConsumer, KafkaProducer
    >>> import time
    >>> import json
    >>> from skmultiflow.data.source.kafka_data_source import KafkaDataSource
    >>> from skmultiflow.data.observer.buffer_data_event_observer import BufferDataEventObserver
    >>>
    >>>
    >>> def record_to_dictionary(record):
    >>>     if record is None:
    >>>         return None
    >>>     return record # already deserialized in consumer
    >>>
    >>>
    >>> # Setup kafka
    >>> broker = "kafkacontainer:9092"
    >>> topic = "scikit-multiflow"
    >>> producer = KafkaProducer(bootstrap_servers=broker)
    >>> consumer = KafkaConsumer(topic, bootstrap_servers=[broker], auto_offset_reset='earliest', enable_auto_commit=True, group_id="test", value_deserializer=lambda x: json.loads(x.decode('utf-8')))
    >>>
    >>> # Setup an event observer and a data source
    >>> buffer_data_event_observer = BufferDataEventObserver()
    >>> data_source = KafkaDataSource(record_to_dictionary, [buffer_data_event_observer], consumer)
    >>>
    >>> # Producer sends a message
    >>> instance_one_msg = json.dumps({'X': [1.0, 2.0, 3.0, 4.0], 'y': ['Iris-setosa']})
    >>> producer.send(topic, instance_one_msg.encode('utf-8'))
    >>> producer.flush()
    >>>
    >>> # Wait until the message is received
    >>> data_source.listen_for_events()
    >>> while(len(buffer_data_event_observer.get_buffer())==0):
    >>>     time.sleep(0.100) # 100ms
    >>>
    >>> first_event = buffer_data_event_observer.get_buffer()[0]
    >>> print('First event: X: {}, y: {}'.format(first_event['X'], first_event['y']))

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
