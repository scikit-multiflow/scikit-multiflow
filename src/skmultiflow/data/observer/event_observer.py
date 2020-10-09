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
        self.result_buffer_ytrue = []
        self.result_buffer_ypred = []
        self.event_buffer_t = []
        self.event_buffer_x = []
        self.event_buffer_y = []

    def update(self, event):
        """For each new event, we follow this sequence:
           - update the train_eval_trigger, so it gains new context
           - check if we shall fit the algorithm with buffered data
             - update the buffer, according to the trigger policy
           - shall predict?
           - shall buffer new instance?
           - get instances to fit from buffer
           - update the buffer
        """
        if event is not None:
            algorithm_type = self.algorithm.algorithm_type()
            if 't' in event:
                t = event['t']
            else:
                t = datetime.datetime.now()
            x = event['X']

            y_true = None
            if algorithm_type == 'CLASSIFICATION' or algorithm_type == 'REGRESSION':
                y_true = event['y']

            t_array = np.array(t)
            x_array = np.array(x)
            y_array = np.array(y_true)

            self.train_eval_trigger.update(event)

            bx_to_fit, by_to_fit = self.train_eval_trigger.instances_to_fit(self.event_buffer_t, self.event_buffer_x, self.event_buffer_y)
            for idx in range(len(bx_to_fit)):
                self.fit_algorithm(bx_to_fit[idx], by_to_fit[idx])

            self.event_buffer_t, self.event_buffer_x, self.event_buffer_y = self.train_eval_trigger.remaining_buffer(self.event_buffer_t, self.event_buffer_x, self.event_buffer_y)

            if self.train_eval_trigger.shall_predict():
                y_pred = self.algorithm.predict(x_array)
                self.result_buffer_ytrue.append(y_true)
                self.result_buffer_ypred.append(y_pred)

            if self.train_eval_trigger.shall_buffer():
                self.event_buffer_t.append(t_array)
                self.event_buffer_x.append(x_array)
                self.event_buffer_y.append(y_array)

            bx_to_fit, by_to_fit = self.train_eval_trigger.instances_to_fit(self.event_buffer_t, self.event_buffer_x, self.event_buffer_y)
            for idx in range(len(bx_to_fit)):
                self.fit_algorithm(bx_to_fit[idx], by_to_fit[idx])

            self.event_buffer_t, self.event_buffer_x, self.event_buffer_y = self.train_eval_trigger.remaining_buffer(self.event_buffer_t, self.event_buffer_x, self.event_buffer_y)

    def fit_algorithm(self, x, y):
        if len(self.result_buffer_ytrue) > 0:
            for result_observer in self.results_observers:
                result_observer.report(self.result_buffer_ypred, self.result_buffer_ytrue)
            self.result_buffer_ypred = []
            self.result_buffer_ytrue = []

        if self.expected_target_values is not None:
            self.algorithm.partial_fit(x, y, self.expected_target_values)
        else:
            self.algorithm.partial_fit(x, y)


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
