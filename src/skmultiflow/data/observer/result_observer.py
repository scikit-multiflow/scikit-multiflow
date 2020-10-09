from abc import ABCMeta, abstractmethod


class ResultObserver(metaclass=ABCMeta):

    @abstractmethod
    def report(self, id, y_pred, y_true):
        raise NotImplementedError


class MetricsResultObserver(ResultObserver):

    def __init__(self, measurements, metrics_reporter):
        self.measurements = measurements
        self.metrics_reporter = metrics_reporter

    def report(self, id, y_pred, y_true):
        for i in range(len(y_true)):
            self.measurements.add_result(y_true[i], y_pred[i])
        self.metrics_reporter.report(self.measurements)


class CSVResultObserver(ResultObserver):

    def __init__(self, filename):
        self.file_handle = open(filename, "w")
        self.file_handle.write("y_pred,y_true\n")
        self.file_handle.flush()

    def report(self, id, y_pred, y_true):
        for i in range(len(id)):
            self.file_handle.write("{},{},{}\n".format(id[i], self.unpack(y_true[i]), self.unpack(y_pred[i])))
            self.file_handle.flush()

    def unpack(self, value):
        if len(value.shape) == 2:
            return value[0][0]
        return value[0]
