from skmultiflow.data.observer.result_observer import ResultObserver
from mockito import kwargs, verify, when, mock
from mockito.matchers import any
import numpy as np


def test_buffered_result_observer():
    metrics = mock()
    reporter = mock()

    when(metrics).get_accuracy().thenReturn(0.9)
    when(reporter).report(any, any).thenReturn(metrics.get_accuracy())
    buffered_result_observer = ResultObserver(metrics, reporter)

    y_true = np.concatenate((np.ones(85), np.zeros(10), np.ones(5)))
    y_pred = np.concatenate((np.ones(90), np.zeros(10)))

    buffered_result_observer.report(y_pred, y_true)

    verify(metrics).get_accuracy()
