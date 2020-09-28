class ResultObserver():
    """ BufferResultObserver class.
    """

    def __init__(self, measurements, reporter):
        """ BufferResultObserver class constructor."""
        self.measurements = measurements
        self.reporter = reporter

    def report(self, y_pred, y_true):
        for i in range(len(y_true)):
            self.measurements.add_result(y_true[i], y_pred[i])
        self.reporter(self.measurements)
