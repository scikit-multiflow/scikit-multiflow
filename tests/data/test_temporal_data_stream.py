from skmultiflow.data.observer.buffer_data_event_observer import BufferDataEventObserver
from skmultiflow.data.observer.buffered_metrics_reporter import BufferedMetricsReporter
from skmultiflow.metrics.measure_collection import ClassificationMeasurements
from skmultiflow.data.observer.prequential_trigger import PrequentialTrigger
from skmultiflow.data.source.dataframe_data_source import DataframeDataSource
from skmultiflow.data.observer.result_observer import ResultObserver
import os
import numpy as np
import pandas as pd
import random
import pytest
from mockito import kwargs, verify, when, mock
from mockito.matchers import any
import time
import numpy as np

def record_to_dictionary(record):
    return {'X': record[["attrib1", "attrib2", "attrib3"]].values,
            'y': record["class"],
            'event_occurrence_time': record["event_occurrence_time"],
            'event_labeling_time': record["event_labeling_delay"]}


def retrieve_metrics(measurements):
    return {'accuracy': measurements.get_accuracy(), 'kappa': measurements.get_kappa()}


def test_temporal_data_stream_time_with_delay(test_path):
    test_file = os.path.join(test_path, 'sea_stream_file_temporal.csv')
    df = pd.read_csv(test_file)
    observer = BufferDataEventObserver()

    data_source = DataframeDataSource(record_to_dictionary, [observer], df)
    data_source.listen_for_events()

    while(len(observer.get_buffer())<1):
        time.sleep(0.100) # 100ms

    first_item = observer.get_buffer()[0]

    np.testing.assert_array_equal(first_item['X'], np.array([0.080429, 8.397186999999999, 7.074928]))
    assert first_item['y'] == 0
    assert first_item['event_occurrence_time'] == '2020-05-12 05:27:05.432909'
    assert first_item['event_labeling_time'] == '2020-05-16 05:27:05.432909'
