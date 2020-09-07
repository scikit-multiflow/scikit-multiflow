from skmultiflow.data.observer.train_eval_trigger import TrainEvalTrigger

class TimeBasedPrequentialTrigger(TrainEvalTrigger):
    """ TimeBasedPrequentialTrigger class.
    """

    def __init__(self, wait_to_test_time_window, time_between, get_event_time):
        """ TimeBasedPrequentialTrigger class constructor."""
        super().__init__()
        self.initial_time_window = None
        self.initialization_period = True
        self.wait_to_test_time_window = wait_to_test_time_window
        self.time_between = time_between
        self.get_event_time = get_event_time
        self.reference_time = None

    def shall_fit(self, event):
        # Once we got the initial window, we shall not fit, but wait to get whole test period
        # Then we evaluate the whole test period, and fit the algorithm with it
        event_time = self.get_event_time(event)
        if self.initialization_period:
            if self.reference_time + self.initial_time_window < self.initialization_period:
                return True
            else:
                self.initialization_period = False
                self.reference_time = event_time
                return False
        return True

    def shall_predict(self, event):
        return not self.initialization_period
