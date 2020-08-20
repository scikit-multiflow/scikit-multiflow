

class QuantityBasedHoldout():
    """ QuantityBasedHoldout class.
    """

    #TODO: support dynamic and static test set
    def __init__(self, n_wait, test_size):
        """ QuantityBasedHoldout class constructor."""
        super().__init__()
        self.n_wait = n_wait
        self.test_size = test_size
        self.wait_counter = 0
        self.test_counter = test_size

    def update(self, algorithm, event):
        algorithm_type = algorithm.algorithm_type()
        x = event['X']
        y_true = None
        if algorithm_type == 'CLASSIFICATION' or algorithm_type == 'REGRESSION':
            y_true = event['y']

        if self.wait_counter < self.n_wait:
            self.wait_counter += 1
            algorithm.partial_fit(x, y_true)
        else:
            self.test_counter = 0

        if self.test_counter < self.test_size:
            self.test_counter += 1
            return algorithm.predict(x), y_true

        if self.test_counter < self.test_size:
            self.wait_counter = 0
