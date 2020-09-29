from skmultiflow.data.observer.metrics_reporter import MetricsReporter


class BufferedMetricsReporter(MetricsReporter):

    def __init__(self, retrieve_metrics):
        self.retrieve_metrics = retrieve_metrics
        self.buffer = None

    def report(self, measurements):
        self.buffer = self.retrieve_metrics(measurements)

    def get_buffer(self):
        return self.buffer

    def get_name(self):
        return "BufferedMetricsReporter"
