from skmultiflow.data.observer.event_observer import EventObserver
import numpy as np

class EvaluationEventObserver(EventObserver):
    """ EvaluationEventObserver class.
    """

    def __init__(self, algorithm, train_eval_trigger, results_observer, expected_target_values=None):
        """ EvaluationEventObserver class constructor."""
        super().__init__()
        self.expected_target_values = expected_target_values
        self.train_eval_trigger = train_eval_trigger
        self.results_observer = results_observer
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
                    self.results_observer.report(self.y_pred_buffer, self.y_true_buffer)
                    self.y_pred_buffer = []
                    self.y_true_buffer = []

                if(self.expected_target_values is not None):
                    self.algorithm.partial_fit(x_array, y_array, self.expected_target_values)
                else:
                    self.algorithm.partial_fit(x_array, y_array)
