from kafka import SimpleProducer, KafkaClient, KafkaConsumer, KafkaProducer
import time
import json
import numpy as np
from skmultiflow.data.source.kafka_data_source import KafkaDataSource
from skmultiflow.data.observer.buffer_data_event_observer import BufferDataEventObserver


def record_to_dictionary(record):
    if record is None:
        return None
    return record # already deserialized in consumer


def test_kafka_data_source(test_path):
    broker = "kafkacontainer:9092"
    topic = "scikit-multiflow"
    producer = KafkaProducer(bootstrap_servers=broker)
    consumer = KafkaConsumer(topic, bootstrap_servers=[broker], auto_offset_reset='earliest', enable_auto_commit=True, group_id="test", value_deserializer=lambda x: json.loads(x.decode('utf-8')))

    buffer_data_event_observer = BufferDataEventObserver()
    data_source = KafkaDataSource(record_to_dictionary, [buffer_data_event_observer], consumer)

    instance_one_msg = json.dumps({'X': [1.0, 2.0, 3.0, 4.0], 'y': ['Iris-setosa']})

    producer.send(topic, instance_one_msg.encode('utf-8'))
    producer.flush()

    data_source.listen_for_events()
    while(len(buffer_data_event_observer.get_buffer())==0):
        time.sleep(0.100) # 100ms

    first_event = buffer_data_event_observer.get_buffer()[0]

    assert np.array_equal(first_event['X'], [1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(first_event['y'], ['Iris-setosa'])

    expected_info = "KafkaDataSource: bootstrap_servers: ['kafkacontainer:9092']; group_id: test"
    assert data_source.get_info() == expected_info
