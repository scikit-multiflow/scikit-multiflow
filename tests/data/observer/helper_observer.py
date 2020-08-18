from skmultiflow.data.observer.event_observer import EventObserver


class PrintlnEventObserver(EventObserver):

    def update(self, event):
        print(event)