from abc import ABCMeta, abstractmethod


class TrainEvalTrigger(metaclass=ABCMeta):
    """ TrainEvalTrigger class.

    This abstract class defines the minimum requirements of a trigger.
    It provides an interface to define criteria when data shall be fitted and evaluated.

    Raises
    ------
    NotImplementedError: This is an abstract class.

    """

    @abstractmethod
    def update(self, event):
        """
        This method aims to store the event, so that can be used to fit or predict
        :param event:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def shall_fit(self):
        raise NotImplementedError

    @abstractmethod
    def shall_predict(self):
        raise NotImplementedError


class PrequentialTrigger(TrainEvalTrigger):
    """ PrequentialTrigger class.
    """

    def __init__(self, n_wait_to_fit):
        """ QuantityBasedHoldout class constructor."""
        super().__init__()
        self.first_time_wait = n_wait_to_fit
        self.first_time_wait_counter = 0

    def update(self, event):
        if self.first_time_wait_counter <= self.first_time_wait:
            self.first_time_wait_counter += 1

    def shall_predict(self):
        if self.first_time_wait_counter <= self.first_time_wait:
            return False
        return True

    def shall_fit(self):
        return True

class TimeBasedHoldoutTrigger(TrainEvalTrigger):
    """ TimeBasedHoldoutTrigger class.
    """

    def __init__(self, initial_time_window, wait_to_test_time_window, test_time_window, get_event_time):
        """ TimeBasedHoldoutTrigger class constructor."""
        super().__init__()
        self.initial_time_window = initial_time_window
        self.wait_to_test_time_window = wait_to_test_time_window
        self.test_time_window = test_time_window
        self.get_event_time = get_event_time

        self.test_mode = False
        self.reference_time = None
        self.target_window = self.initial_time_window


    def update(self, event):
        event_time = self.get_event_time(event)
        if self.reference_time is None:
            self.reference_time = event_time

        time_between = event_time - self.reference_time
        if time_between > self.target_window:
            self.reference_time = event_time
            print("Switched to reference time: {}".format(self.reference_time))
            self.test_mode = not self.test_mode
            if self.test_mode:
                self.target_window = self.test_time_window
            else:
                self.target_window = self.wait_to_test_time_window

    def shall_fit(self):
        return not self.test_mode

    def shall_predict(self):
        return self.test_mode

class QuantityBasedHoldoutTrigger(TrainEvalTrigger):
    """ QuantityBasedHoldout class.
    """

    #TODO: support dynamic and static test set: we support it considering the report evaluation policy
    def __init__(self, first_time_wait, n_wait_to_test, test_size):
        """ QuantityBasedHoldout class constructor."""
        super().__init__()
        self.first_time_wait = max(first_time_wait, n_wait_to_test)
        self.n_wait_to_test = n_wait_to_test
        self.test_size = test_size
        self.test_mode = False
        self.events_counter = 0
        self.events_target = self.first_time_wait

    def update(self, event):
        if self.events_counter == self.events_target:
            self.test_mode = not self.test_mode
            self.events_counter = 0
            if self.test_mode:
                self.events_target = self.test_size
            else:
                self.events_target = self.n_wait_to_test

        if self.events_counter < self.events_target:
            self.events_counter += 1


    def shall_fit(self):
        return not self.test_mode

    def shall_predict(self):
        return self.test_mode
