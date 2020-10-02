from skmultiflow.data.observer.event_observer import EventObserver
import datetime


class StreamSpeedObserver(EventObserver):
    """Supports same functionality originaly envisioned in EvaluateStreamGenerationSpeed
        while providing greater flexibility on how we deal with streams as well
        as metrics we report"""

    def __init__(self, last_n_samples):
        self.last_n_samples = last_n_samples
        self.buffer = []

    def update(self, event):
        if(event is not None):
            self.buffer.append(datetime.datetime.now())
            self.buffer = self.buffer[-self.last_n_samples:]

    def get_buffer(self):
        return self.buffer

    def get_name(self):
        return "StreamSpeedObserver"
