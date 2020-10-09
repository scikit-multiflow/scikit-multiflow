from abc import ABCMeta, abstractmethod


class ResultObserver(metaclass=ABCMeta):

    @abstractmethod
    def report(self, y_pred, y_true):
        raise NotImplementedError


class MetricsResultObserver(ResultObserver):

    def __init__(self, measurements, metrics_reporter):
        self.measurements = measurements
        self.metrics_reporter = metrics_reporter

    def report(self, y_pred, y_true):
        for i in range(len(y_true)):
            self.measurements.add_result(y_true[i], y_pred[i])
        self.metrics_reporter.report(self.measurements)


class CSVResultObserver(ResultObserver):

    def __init__(self, filename):
        self.file_handle = open(filename, "w")
        self.file_handle.write("y_pred,y_true")
        self.file_handle.flush()

    def report(self, y_pred, y_true):
        for i in range(len(y_true)):
            self.file_handle.write("{},{}".format(y_true[i], y_pred[i]))
            self.file_handle.flush()

