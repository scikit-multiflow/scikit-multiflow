from skmultiflow.data.observer.event_observer import EventObserver


class HoldoutEvaluationEventObserver(EventObserver):
    """ HoldoutEvaluationEventObserver class.
    """

    def __init__(self, algorithm, holdout_strategy, results_observer):
        """ AlgorithmEventObserver class constructor."""
        super().__init__()
        self.algorithm = algorithm
        self.holdout_strategy = holdout_strategy
        self.results_observer = results_observer

    def update(self, event):
        y_pred, y_true = self.holdout_strategy.process(self.algorithm, event)
        if y_pred is not None:
            self.results_observer.report(y_pred, y_true)
