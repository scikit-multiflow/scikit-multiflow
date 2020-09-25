from skmultiflow.data.observer.quantity_based_holdout_trigger import QuantityBasedHoldoutTrigger
from skmultiflow.data.observer.evaluation_event_observer import EvaluationEventObserver
from skmultiflow.data.source.file_data_source import FileDataSource
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier

from mockito import kwargs, verify, when, mock
from mockito.matchers import any

import numpy as np
import time
import os

def one_hot_encoding(string_value):
    if(string_value == 'Iris-setosa'):
        return 0
    if(string_value == 'Iris-versicolor'):
        return 1
    if(string_value == 'Iris-virginica'):
        return 2
    return -1

def record_to_dictionary(record):
    record_array = record.strip().split(',')
    if(len(record_array)==5):
        return {'X': np.array([float(record_array[0]), float(record_array[1]), float(record_array[2]), float(record_array[3])]), 'y': [one_hot_encoding(record_array[4])]}
    return None


def test_evaluation_event_observer(test_path):
    test_file = os.path.join(test_path, 'iris.data')
    train_eval_trigger = QuantityBasedHoldoutTrigger(5, 10, 20)
    algorithm = HoeffdingAdaptiveTreeClassifier(leaf_prediction='mc', random_state=1)

    results_observer = mock()
    evaluation_event_observer = EvaluationEventObserver(algorithm, train_eval_trigger, results_observer, [0, 1, 2])

    data_source = FileDataSource(record_to_dictionary, [evaluation_event_observer], test_file)
    data_source.listen_for_events()
    time.sleep(3)

    verify(results_observer, times=4).report(any, any)

