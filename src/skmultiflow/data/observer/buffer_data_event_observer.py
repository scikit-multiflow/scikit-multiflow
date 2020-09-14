from skmultiflow.data.observer.event_observer import EventObserver


class BufferDataEventObserver(EventObserver):

    def __init__(self):
        self.buffer = []

    def update(self, event):
        if(event is not None):
            self.buffer.append(event)

    def get_buffer(self):
        return self.buffer

    def get_name(self):
        return "BufferDataEventObserver"
