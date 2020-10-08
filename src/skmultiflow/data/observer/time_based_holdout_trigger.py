from skmultiflow.data.observer.train_eval_trigger import TrainEvalTrigger

class TimeBasedHoldoutTrigger(TrainEvalTrigger):
    """ TimeBasedHoldoutTrigger class.
    """

    def __init__(self, initial_time_window, wait_to_test_time_window, test_time_window, get_event_time):
        """ TimeBasedHoldoutTrigger class constructor."""
        super().__init__()
        self.initial_time_window = initial_time_window
        self.wait_to_test_time_window = wait_to_test_time_window
        self.test_time_window = test_time_window
        self.get_event_time = get_event_time

        self.test_mode = False
        self.reference_time = None
        self.target_window = self.initial_time_window


    def update(self, event):
        event_time = self.get_event_time(event)
        if self.reference_time is None:
            self.reference_time = event_time

        time_between = event_time - self.reference_time
        if time_between > self.target_window:
            self.reference_time = event_time
            print("Switched to reference time: {}".format(self.reference_time))
            self.test_mode = not self.test_mode
            if self.test_mode:
                self.target_window = self.test_time_window
            else:
                self.target_window = self.wait_to_test_time_window

    def shall_fit(self):
        return not self.test_mode

    def shall_predict(self):
        return self.test_mode
