import bisect


class QuantityBasedBuffer:
    def __init__(self, n):
        self.n = n
        self.buffer = []

    def register_value(self, value):
        self.buffer.append(value)
        self.buffer = self.buffer[-self.n:]
        return self.buffer


class TimeBasedBuffer:
    def __init__(self, time_window):
        self.time_window = time_window
        self.buffer = []

    def register_value(self, tuple):
        """tuple: (date_time, value)"""
        bisect.insort(self.buffer, tuple)
        while self.buffer[0][0]-self.buffer[-1][0] > self.time_window:
            self.buffer.pop(0)
        return [ x[1] for x in self.buffer]
