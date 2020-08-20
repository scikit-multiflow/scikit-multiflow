from skmultiflow.data.observer.event_observer import EventObserver


class PrequentialEvaluationEventObserver(EventObserver):
    """ PrequentialEvaluationEventObserver class.
    """

    def __init__(self, results_observer):
        """ AlgorithmEventObserver class constructor."""
        super().__init__()
        self.results_observer = results_observer

    def update(self, event):
        algorithm_type = self.algorithm.algorithm_type()
        x = event['X']
        y_true = None
        if algorithm_type == 'CLASSIFICATION' or algorithm_type == 'REGRESSION':
            y_true = event['y']

        y_pred = self.algorithm.predict(x)
        self.results_observer.report(y_pred, y_true)
        self.algorithm.partial_fit(x, y_true)
