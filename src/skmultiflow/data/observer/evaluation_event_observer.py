from skmultiflow.data.observer.event_observer import EventObserver
import numpy as np

class EvaluationEventObserver(EventObserver):
    """ EvaluationEventObserver class.
    """

    def __init__(self, algorithm, train_eval_trigger, results_observer):
        """ EvaluationEventObserver class constructor."""
        super().__init__()
        self.algorithm = algorithm
        self.train_eval_trigger = train_eval_trigger
        self.results_observer = results_observer
        self.y_true_buffer = []
        self.y_pred_buffer = []
        self.x_buffer = []

    def update(self, event):
        algorithm_type = self.algorithm.algorithm_type()
        x = event['X']
        y_true = None
        if algorithm_type == 'CLASSIFICATION' or algorithm_type == 'REGRESSION':
            y_true = event['y']

        if self.train_eval_trigger.shall_predict(event):
            y_pred = self.algorithm.predict(x)
            self.y_true_buffer.append(y_true)
            self.y_pred_buffer.append(y_pred)
            self.x_buffer.append(x)

        if self.train_eval_trigger.shall_fit(event):
            self.results_observer.report(self.y_pred_buffer, self.y_true_buffer)
            for idx in range(0, len(self.x_buffer)):
                x_array = np.array(self.x_buffer[idx])
                x_array = x_array.reshape(x_array.shape[0], 1)
                y_array = np.array(0)
                y_array = y_array.reshape(1, 1)
                print("selected: {}".format(len(self.y_true_buffer)))
                print("selected: {}".format(self.y_true_buffer))
                self.algorithm.partial_fit(x_array, y_array)

            self.y_pred_buffer = []
            self.y_true_buffer = []
            self.x_buffer = []
