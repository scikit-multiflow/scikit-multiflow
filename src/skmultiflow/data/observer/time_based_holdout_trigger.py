
class TimeBasedPrequentialTrigger():
    """ TimeBasedPrequentialTrigger class.
    """

    def __init__(self, wait_to_test_time_window):
        """ TimeBasedHoldoutTrigger class constructor."""
        super().__init__()
        self.initial_time_window = None
        self.initialization_period = True
        self.wait_to_test_time_window = wait_to_test_time_window
        self.reference_time = None

    def shall_fit(self, event):
        # TODO: once we got the initial window, we shall not fit, but wait to get whole test period
        # Then we evaluate the whole test period, and fit the algorithm with it
        if self.initialization_period:
            if self.reference_time + self.initial_time_window < self.initialization_period:
                return True
            else:
                self.initialization_period = False
                self.reference_time = event[''] #TODO retrieve time of interest
                return False
        event_time = event['']#TODO retrieve time of interest
        if (event_time - self.reference_time) < self.wait_to_test_time_window:
            return False
        return True

    def shall_predict(self, event):
        event_time = event['']#TODO retrieve time of interest
        if (event_time - self.reference_time) < self.wait_to_test_time_window:
            return False
        else:
            self.reference_time = event_time
            return True
