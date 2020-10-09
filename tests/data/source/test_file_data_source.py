import os
import time
import numpy as np
from skmultiflow.data.source.file_data_source import FileDataSource
from skmultiflow.data.observer.event_observer import BufferDataEventObserver


def record_to_dictionary(record):
    record_array = record.strip().split(',')
    if(len(record_array)==5):
        return {'X': [record_array[0], record_array[1], record_array[2], record_array[3]], 'y': [record_array[4]]}
    return None


def test_file_data_source(test_path):
    test_file = os.path.join(test_path, 'iris.data')
    buffer_data_event_observer = BufferDataEventObserver()
    data_source = FileDataSource(record_to_dictionary, [buffer_data_event_observer], test_file)
    data_source.listen_for_events()
    while(len(buffer_data_event_observer.get_buffer())<2):
        time.sleep(0.100) # 100ms

    first_event = buffer_data_event_observer.get_buffer()[0]
    second_event = buffer_data_event_observer.get_buffer()[1]

    assert np.array_equal(first_event['X'], ['5.1', '3.5', '1.4', '0.2'])
    assert np.array_equal(second_event['X'], ['4.9', '3.0', '1.4', '0.2'])
    assert np.array_equal(first_event['y'], ['Iris-setosa'])
    assert np.array_equal(second_event['y'], ['Iris-setosa'])

    expected_info = "FileDataSource: iris.data; observers: ['BufferDataEventObserver']"
    assert data_source.get_info() == expected_info
