from abc import ABCMeta, abstractmethod


class MetricsReporter(metaclass=ABCMeta):
    """ MetricsReporter class.

    This abstract class defines the minimum requirements of a metrics reporter.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """

    @abstractmethod
    def report(self, measurements):
        raise NotImplementedError


class BufferedMetricsReporter(MetricsReporter):
    def __init__(self, retrieve_metrics):
        self.retrieve_metrics = retrieve_metrics
        self.buffer = None

    def report(self, measurements):
        self.buffer = self.retrieve_metrics(measurements)

    def get_buffer(self):
        return self.buffer


class CSVMetricsReporter(MetricsReporter):
    def __init__(self, filename, retrieve_metrics):
        self.retrieve_metrics = retrieve_metrics
        self.file_handle = open(filename, "w")
        self.first_time = False

    def report(self, measurements):
        metrics = self.retrieve_metrics(measurements)
        metric_names = sorted(list(self.retrieve_metrics(measurements).keys()))

        if self.first_time:
            # write the file header
            self.file_handle.write(','.join(metric_names))
            self.first_time = False

        values = [str(metrics[name]) for name in metric_names]
        self.file_handle.write(','.join(values))
        self.file_handle.flush()
