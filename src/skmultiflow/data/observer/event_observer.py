from abc import ABCMeta, abstractmethod
import numpy as np
import datetime


class EventObserver(metaclass=ABCMeta):
    """ EventObserver class.

    This abstract class defines the minimum requirements of a data source.
    It provides an interface to access entries, streamed from certain source.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """

    @abstractmethod
    def update(self, event):
        raise NotImplementedError


class BufferDataEventObserver(EventObserver):

    def __init__(self):
        self.buffer = []

    def update(self, event):
        if(event is not None):
            self.buffer.append(event)

    def get_buffer(self):
        return self.buffer

    def get_name(self):
        return "BufferDataEventObserver"


class EvaluationEventObserver(EventObserver):
    """ EvaluationEventObserver class.
    """

    def __init__(self, algorithm, train_eval_trigger, results_observers, expected_target_values=None):
        """ EvaluationEventObserver class constructor."""
        self.expected_target_values = expected_target_values
        self.train_eval_trigger = train_eval_trigger
        self.results_observers = results_observers
        self.algorithm = algorithm
        self.y_true_buffer = []
        self.y_pred_buffer = []
        self.x_buffer = []

    def update(self, event):
        if event is not None:
            algorithm_type = self.algorithm.algorithm_type()
            x = event['X']

            y_true = None
            if algorithm_type == 'CLASSIFICATION' or algorithm_type == 'REGRESSION':
                y_true = event['y']

            x_array = np.array(x)
            y_array = np.array(y_true)

            self.train_eval_trigger.update(event)

            if self.train_eval_trigger.shall_predict():
                y_pred = self.algorithm.predict(x_array)
                self.y_true_buffer.append(y_true)
                self.y_pred_buffer.append(y_pred)

            if self.train_eval_trigger.shall_fit():
                if len(self.y_true_buffer)>0:
                    for result_observer in self.results_observers:
                        result_observer.report(self.y_pred_buffer, self.y_true_buffer)
                    self.y_pred_buffer = []
                    self.y_true_buffer = []

                if self.expected_target_values is not None:
                    self.algorithm.partial_fit(x_array, y_array, self.expected_target_values)
                else:
                    self.algorithm.partial_fit(x_array, y_array)


class StreamSpeedObserver(EventObserver):
    """Supports same functionality originaly envisioned in EvaluateStreamGenerationSpeed
        while providing greater flexibility on how we deal with streams as well
        as metrics we report"""

    def __init__(self, last_n_samples):
        self.last_n_samples = last_n_samples
        self.buffer = []

    def update(self, event):
        if(event is not None):
            self.buffer.append(datetime.datetime.now())
            self.buffer = self.buffer[-self.last_n_samples:]

    def get_buffer(self):
        return self.buffer

    def get_name(self):
        return "StreamSpeedObserver"
