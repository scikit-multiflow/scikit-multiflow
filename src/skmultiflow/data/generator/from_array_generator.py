class FromArrayGenerator():
    """
    Stream provided data over and over again

    Parameters
    ----------


    Notes
    -----
    The data generated corresponds to sine (attribute 1) and cosine
    (attribute 2) functions. Anomalies are induced by replacing values
    from attribute 2 with values from a sine function different to the one
    used in attribute 1. The ``contextual`` flag can be used to introduce
    contextual anomalies which are values in the normal global range,
    but abnormal compared to the seasonal pattern. Contextual attributes
    are introduced by replacing values in attribute 2 with values from
    attribute 1.

    """

    def __init__(self, array, infinite_loop=True):
        self.array = array
        self.infinite_loop = infinite_loop
        self.sample_idx = 0
        self.name = 'FromArrayGenerator'

        self._prepare_for_use()

    def _prepare_for_use(self):
        pass

    def next_sample(self):
        """
        Get the next sample from the stream.

        Parameters
        ----------
        batch_size: int (optional, default=1)
            The number of samples to return.

        Returns
        -------
        tuple of arrays
            Return a tuple with the features X and the target y for
            the batch_size samples that are requested.

        """
        if self.sample_idx < len(self.array):
            nxt_sample = self.array[self.sample_idx]
            self.sample_idx += 1
            if self.sample_idx >= len(self.array):
                if self.infinite_loop:
                    self.sample_idx = 0
            return nxt_sample[0], nxt_sample[1]
        return None

    def get_info(self):
        return "FromArrayGenerator"
