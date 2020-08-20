
class TimeBasedHoldout():
    """ TimeBasedHoldout class.
    """

    def __init__(self, n_wait, test_size):
        """ QuantityBasedHoldout class constructor."""
        super().__init__()
        self.n_wait = n_wait
        self.test_size = test_size
        self.wait_counter = 0
        self.test_counter = test_size

    def update(self, algorithm, event):
        # TODO: get event time and compute time since last test set. Ex.: we evaluate every month?
        return None
