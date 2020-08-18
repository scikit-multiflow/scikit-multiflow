import os
from skmultiflow.data.source.file_data_source import FileDataSource
from skmultiflow.data.observer.helper_observer import PrintlnEventObserver

def record_to_dictionary(record):
    record_array = record.split(',')
    return {'X': [record_array[0], record_array[1], record_array[2]], 'y': [record_array[3]]}

def test_file_stream(test_path):
    test_file = os.path.join(test_path, 'sea_stream_file.csv')
    data_source = FileDataSource(record_to_dictionary, [PrintlnEventObserver()], test_file)

    expected_info = "FileStream(filename='sea_stream_file.csv', " \
                    "target_idx=-1, n_targets=1, cat_features=None)"
    assert  data_source.get_info() == expected_info
