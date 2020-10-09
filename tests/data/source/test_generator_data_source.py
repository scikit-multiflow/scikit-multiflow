from skmultiflow.data.observer.event_observer import BufferDataEventObserver
from skmultiflow.data.generator.anomaly_sine_generator import AnomalySineGenerator
from skmultiflow.data.source.generator_data_source import GeneratorDataSource
import numpy as np
import time


def record_to_dictionary(record):
    if record is None:
        return None
    return record


def test_generator_data_source():
    generator = AnomalySineGenerator(random_state=3)
    buffer_data_event_observer = BufferDataEventObserver()
    data_source = GeneratorDataSource(record_to_dictionary, [buffer_data_event_observer], generator)
    data_source.listen_for_events()

    while (len(buffer_data_event_observer.get_buffer()) < 5):
        time.sleep(0.100)  # 100ms
    events = buffer_data_event_observer.get_buffer()[:5]

    expected = [(np.array([[0.89431424, 2.15223693]]), np.array([1.])), (np.array([[0.46565888, 0.05565128]]), np.array([0.])), (np.array([[0.52767427, 0.45518165]]), np.array([0.])), (np.array([[-0.25010759, -0.39191752]]), np.array([0.])), (np.array([[0.70277688, 1.11163411]]), np.array([0.]))]

    for j in range(0,5):
        assert np.alltrue(np.isclose(events[j][0], expected[j][0]))
        assert np.alltrue(np.isclose(events[j][1], expected[j][1]))
