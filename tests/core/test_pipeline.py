import os

from skmultiflow.data.observer.evaluation_event_observer import EvaluationEventObserver
from skmultiflow.data.observer.buffered_metrics_reporter import BufferedMetricsReporter
from skmultiflow.metrics.measure_collection import ClassificationMeasurements
from skmultiflow.data.observer.prequential_trigger import PrequentialTrigger
from skmultiflow.data.source.array_data_source import ArrayDataSource
from skmultiflow.data.observer.result_observer import ResultObserver
from skmultiflow.transform import OneHotToCategorical
from skmultiflow.lazy import KNNADWINClassifier
from skmultiflow.core import Pipeline
import numpy as np
import time


def record_to_dictionary(record):
    return {'X': record['X'], 'y': np.array(record['y'])}


def retrieve_metrics(measurements):
    return {'accuracy': measurements.get_accuracy(), 'kappa': measurements.get_kappa()}


def test_pipeline(test_path):
    n_categories = 5

    test_file = os.path.join(test_path, 'data-one-hot.npz')
    data = np.load(test_file)

    data_as_dict = []
    for i in range(0, len(data['X'])):
        data_as_dict.append({'X':data['X'][i].reshape(1, 25), 'y':np.array(data['y'][i]).reshape(1, 1)})


    # Setup transformer
    cat_att_idx = [[i + j for i in range(n_categories)] for j in range(0, n_categories * n_categories, n_categories)]
    transformer = OneHotToCategorical(categorical_list=cat_att_idx)

    # Set up the classifier
    classifier = KNNADWINClassifier(n_neighbors=2, max_window_size=50, leaf_size=40)
    # Setup the pipeline
    pipe = Pipeline([('one-hot', transformer), ('KNNADWINClassifier', classifier)])

    train_eval_trigger = PrequentialTrigger(10)
    reporter = BufferedMetricsReporter(retrieve_metrics)
    results_observer = ResultObserver(ClassificationMeasurements(), reporter)
    evaluation_event_observer = EvaluationEventObserver(pipe, train_eval_trigger, results_observer, [0, 1])

    data_source = ArrayDataSource(record_to_dictionary, [evaluation_event_observer], data_as_dict)

    data_source.listen_for_events()
    time.sleep(3)

    expected_accuracy = 0.5555555555555556
    expected_kappa = 0.11111111111111116

    assert np.isclose(expected_accuracy, reporter.get_buffer()['accuracy'])
    assert np.isclose(expected_kappa, reporter.get_buffer()['kappa'])

