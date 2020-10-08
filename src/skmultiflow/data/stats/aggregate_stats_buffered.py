import numpy as np

class StdDev:
    """
    Taken from
    https://math.stackexchange.com/questions/198336/how-to-calculate-standard-deviation-with-streaming-inputs
    """

    def __init__(self, buffer):
        self.buffer = buffer

    def register_value(self, value):
        values = self.buffer.register_value(value)
        return np.std(np.array(values))


class Median:
    def __init__(self, buffer):
        self.buffer = buffer

    def register_value(self, value):
        values = self.buffer.register_value(value)
        return np.median(np.array(values))


class Mean:
    def __init__(self, buffer):
        self.buffer = buffer

    def register_value(self, value):
        values = self.buffer.register_value(value)
        return np.mean(np.array(values))
