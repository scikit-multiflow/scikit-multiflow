from skmultiflow.data.observer.quantity_based_holdout_trigger import QuantityBasedHoldoutTrigger
from skmultiflow.data.observer.buffer_data_event_observer import BufferDataEventObserver
from skmultiflow.data.observer.evaluation_event_observer import EvaluationEventObserver
from skmultiflow.data.observer.buffer_result_observer import BufferResultObserver
from skmultiflow.data.source.file_data_source import FileDataSource
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
import numpy as np
import time
import os

def one_hot_encoding(string_value):
    if(string_value == 'Iris-setosa'):
        return 0
    return 1

def record_to_dictionary(record):
    record_array = record.strip().split(',')
    if(len(record_array)==5):
        return {'X': np.array([float(record_array[0]), float(record_array[1]), float(record_array[2]), float(record_array[3])]), 'y': [one_hot_encoding(record_array[4])]}
    return None


def test_file_data_source(test_path):
    test_file = os.path.join(test_path, 'iris.data')
    train_eval_trigger = QuantityBasedHoldoutTrigger(5, 10, 20)
    algorithm = HoeffdingAdaptiveTreeClassifier(leaf_prediction='mc', random_state=1)
    buffer_data_event_observer = BufferDataEventObserver()
    results_observer = BufferResultObserver()
    evaluation_event_observer = EvaluationEventObserver(algorithm, train_eval_trigger, results_observer)
    data_source = FileDataSource(record_to_dictionary, [evaluation_event_observer, buffer_data_event_observer], test_file)
    data_source.listen_for_events()

    time.sleep(10)
    print("Will exit now :)")


