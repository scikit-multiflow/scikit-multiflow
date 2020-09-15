from skmultiflow.data.observer.train_eval_trigger import TrainEvalTrigger

class QuantityBasedHoldoutTrigger(TrainEvalTrigger):
    """ QuantityBasedHoldout class.
    """

    #TODO: support dynamic and static test set: we support it considering the report evaluation policy
    def __init__(self, n_wait_to_test, test_size):
        """ QuantityBasedHoldout class constructor."""
        super().__init__()
        self.n_wait_to_test = n_wait_to_test
        self.test_size = test_size
        self.wait_to_test_counter = 0
        self.test_counter = test_size

    def shall_fit(self, event):
        if self.wait_to_test_counter < self.n_wait_to_test:
            self.wait_to_test_counter += 1
            return True
        else:
            self.test_counter = 0
        return False

    def shall_predict(self, event):
        response = False
        if self.test_counter < self.test_size:
            self.test_counter += 1
            response = True
        if self.test_counter < self.test_size:
            self.wait_to_test_counter = 0

        return response
