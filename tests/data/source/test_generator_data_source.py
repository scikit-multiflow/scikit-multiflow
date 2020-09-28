import os
import time
import numpy as np
from skmultiflow.data.generator.sea_generator import SEAGenerator
from skmultiflow.data.source.generator_data_source import GeneratorDataSource
from skmultiflow.data.observer.buffer_data_event_observer import BufferDataEventObserver


def record_to_dictionary(record):
    if record is None:
        return None
    return record


def test_generator_data_source(test_path):
    sea_generator = SEAGenerator(classification_function=2, random_state=112, balance_classes=False,
                                 noise_percentage=0.28)
    buffer_data_event_observer = BufferDataEventObserver()
    data_source = GeneratorDataSource(record_to_dictionary, [buffer_data_event_observer], sea_generator)
    data_source.listen_for_events()

    while (len(buffer_data_event_observer.get_buffer()) < 10):
        time.sleep(0.100)  # 100ms
    events = buffer_data_event_observer.get_buffer()[:10]

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'sea_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    for j in range(0,10):
        assert np.alltrue(np.isclose(events[j]['X'], X_expected[j]))
        assert np.alltrue(np.isclose(events[j]['y'], y_expected[j]))
