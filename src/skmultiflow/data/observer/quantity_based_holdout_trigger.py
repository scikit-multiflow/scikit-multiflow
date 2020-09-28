from skmultiflow.data.observer.train_eval_trigger import TrainEvalTrigger

class QuantityBasedHoldoutTrigger(TrainEvalTrigger):
    """ QuantityBasedHoldout class.
    """

    #TODO: support dynamic and static test set: we support it considering the report evaluation policy
    def __init__(self, first_time_wait, n_wait_to_test, test_size):
        """ QuantityBasedHoldout class constructor."""
        super().__init__()
        self.first_time_wait = max(first_time_wait, n_wait_to_test)
        self.n_wait_to_test = n_wait_to_test
        self.test_size = test_size
        self.test_mode = False
        self.events_counter = 0
        self.events_target = self.first_time_wait

    def update(self, event):
        if self.events_counter == self.events_target:
            self.test_mode = not self.test_mode
            self.events_counter = 0
            if self.test_mode:
                self.events_target = self.test_size
            else:
                self.events_target = self.n_wait_to_test

        if self.events_counter < self.events_target:
            self.events_counter += 1


    def shall_fit(self):
        return not self.test_mode

    def shall_predict(self):
        return self.test_mode
