from skmultiflow.data.observer.train_eval_trigger import TrainEvalTrigger

class QuantityBasedPrequentialTrigger(TrainEvalTrigger):
    """ QuantityBasedPrequentialTrigger class.
    """

    def __init__(self, n_wait_to_fit):
        """ QuantityBasedHoldout class constructor."""
        super().__init__()
        self.n_wait_to_fit = n_wait_to_fit
        self.wait_to_fit_counter = 0

    def update(self, event):
        # TODO: complete
        return True

    def shall_predict(self, event):
        if self.wait_to_fit_counter < self.n_wait_to_fit:
            self.wait_to_fit_counter += 1
        return True

    def shall_fit(self, event):
        if self.wait_to_fit_counter >= self.n_wait_to_fit:
            self.wait_to_fit_counter = 0
            return True
        return False
