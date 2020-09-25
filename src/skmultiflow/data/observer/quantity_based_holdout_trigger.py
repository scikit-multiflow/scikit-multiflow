from skmultiflow.data.observer.train_eval_trigger import TrainEvalTrigger

class QuantityBasedHoldoutTrigger(TrainEvalTrigger):
    """ QuantityBasedHoldout class.
    """

    #TODO: support dynamic and static test set: we support it considering the report evaluation policy
    def __init__(self, first_time_wait, n_wait_to_test, test_size):
        """ QuantityBasedHoldout class constructor."""
        super().__init__()
        self.first_time_wait = first_time_wait
        self.first_time_wait_counter = 0
        self.n_wait_to_test = n_wait_to_test
        self.test_size = test_size
        self.wait_to_test_counter = 0
        self.test_cases_counter = 0

    def update(self, event):
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
        if self.first_time_wait_counter < self.first_time_wait:
            return True
        else:
            return self.wait_to_test_counter < self.n_wait_to_test and self.test_cases_counter == 0

    def shall_predict(self):
        if self.first_time_wait_counter < self.first_time_wait:
            return False
        else:
            return self.test_cases_counter == self.test_size

