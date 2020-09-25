from skmultiflow.data.observer.train_eval_trigger import TrainEvalTrigger

class PrequentialTrigger(TrainEvalTrigger):
    """ PrequentialTrigger class.
    """

    def __init__(self, n_wait_to_fit):
        """ QuantityBasedHoldout class constructor."""
        super().__init__()
        self.first_time_wait = n_wait_to_fit
        self.first_time_wait_counter = 0

    def update(self, event):
        if self.first_time_wait_counter <= self.first_time_wait:
            self.first_time_wait_counter += 1

    def shall_predict(self):
        if self.first_time_wait_counter <= self.first_time_wait:
            return False
        return True

    def shall_fit(self):
        return True
