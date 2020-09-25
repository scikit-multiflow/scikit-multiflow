from skmultiflow.data.observer.train_eval_trigger import TrainEvalTrigger

class TimeBasedHoldoutTrigger(TrainEvalTrigger):
    """ TimeBasedHoldoutTrigger class.
    """

    def __init__(self, wait_to_test_time_window, time_between, get_event_time):
        """ TimeBasedHoldoutTrigger class constructor."""
        super().__init__()
        self.initial_time_window = None
        self.initialization_period = True
        self.wait_to_test_time_window = wait_to_test_time_window
        self.time_between = time_between
        self.get_event_time = get_event_time
        self.reference_time = None

    def update(self, event):
        event_time = self.get_event_time(event)
        if self.reference_time is None:
            self.reference_time = event_time


        if self.first_time_wait_counter < self.first_time_wait:
            self.first_time_wait_counter += 1
        else:
            if self.wait_to_test_counter == 0 and self.test_cases_counter == self.test_size:
                self.test_cases_counter = 0
            if self.wait_to_test_counter < self.n_wait_to_test:
                self.wait_to_test_counter += 1
            else:
                self.test_cases_counter += 1

            if self.test_cases_counter == self.test_size:
                self.wait_to_test_counter = 0

    def shall_fit(self):
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
        if self.time_between(self.reference_time, self.get_event_time(event)) < self.wait_to_test_time_window:
            return False
        return True

    def shall_predict(self):
        event_time = self.get_event_time(event)
        if self.time_between(self.reference_time, self.get_event_time(event)) < self.wait_to_test_time_window:
            return False
        else:
            self.reference_time = event_time
            return True
