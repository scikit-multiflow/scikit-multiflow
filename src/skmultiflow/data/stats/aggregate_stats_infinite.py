from tdigest import TDigest


class StdDev:
    """
    Taken from
    https://math.stackexchange.com/questions/198336/how-to-calculate-standard-deviation-with-streaming-inputs
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def register_value(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta/self.n
        self.M2 += delta*(value - self.mean)
        if self.n < 2:
            return float('nan')
        else:
            return self.M2 / (self.n - 1)


class Median:
    def __init__(self):
        self.tdigest = TDigest()

    def register_value(self, value):
        self.tdigest.update(value)
        return self.tdigest.percentile(50)


class Mean:
    def __init__(self):
        self.tdigest = TDigest()

    def register_value(self, value):
        self.tdigest.update(value)
        return self.tdigest.trimmed_mean(1, 99)

