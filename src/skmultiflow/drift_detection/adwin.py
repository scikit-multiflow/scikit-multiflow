from collections import deque, namedtuple
import numpy as np

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class ADWIN(BaseDriftDetector):
    """ Adaptive Windowing method for concept drift detection.

    Parameters
    ----------
    delta : float (default=0.002)
        The delta parameter for the ADWIN algorithm.

    Notes
    -----
    ADWIN [1]_ (ADaptive WINdowing) is an adaptive sliding window
    algorithm for detecting change and keeping updated statistics about
    a data stream. ADWIN allows algorithms not adapted for drifting
    data to be resistant to this phenomenon.

    The general idea is to keep statistics from a window of variable
    size while detecting concept drift.

    The algorithm will decide the size of the window by cutting
    the statistics' window at different points and analysing
    the average of some statistic over these two windows. If
    the absolute value of the difference between the two averages
    surpasses a pre-defined threshold, change is detected at that point
    and all data before that time is discarded.

    References
    ----------
    .. [1] Bifet, Albert, and Ricard Gavalda.
       "Learning from time-changing data with adaptive windowing."
       In Proceedings of the 2007 SIAM international conference on data
       mining, pp. 443-448.
       Society for Industrial and Applied Mathematics, 2007.

    Examples
    --------
    >>> import numpy as np
    >>> from skmultiflow.drift_detection.adwin import ADWIN
    >>> adwin = ADWIN()
    >>> # Simulating a data stream as a normal distribution of 1's
    ... # and 0's.
    >>> data_stream = np.random.randint(2, size=2000)
    >>> # Changing the data concept from index 999 to 2000.
    >>> for i in range(999, 2000):
    ...     data_stream[i] = np.random.randint(4, high=8)
    >>> # Adding stream elements to ADWIN and verifying if drift
    ... # occurred.
    >>> for i in range(2000):
    ...     adwin.add_element(data_stream[i])
    ...     if adwin.detected_change():
    ...         print('Change detected in data: '
    ...               + str(data_stream[i])
    ...               + ' - at index: '
    ...               + str(i))
    """

    # This is arbitrary & has no impact on the behaviour of ADWIN.
    MAX_BUCKETS = 5
    Bucket = namedtuple('bucket', ['total', 'variance']) # type

    def __init__(self, delta=.002):
        """ ADWIN init.

        The sliding window is stored in `window` as a deque of deque,
        each one keeping a list of buckets of the same size & of
        type `bucket`. The oldest elements are the ones with the
        highest index.
        """
        super().__init__()
        self.delta = delta
        self.window = deque(deque())
        self.total = 0
        self._variance = 0
        self.width = 0

        self.min_window_longitude = 10

        self.time = 0
        self.total_width = 0

        self.detect = 0
        self.n_detections = 0
        self.detect_twice = 0
        self.clock = 32

        self.was_bucket_deleted = False
        self.max_n_buckets = 0
        self.min_window_length = 5
        super().reset()

    def reset(self):
        """ Reset detectors.

        Resets statistics and adwin's window.

        Returns
        -------
        ADWIN
            self
        """
        self.__init__(delta=self.delta)

    def get_change(self):
        """ Get drift.

        Returns
        -------
        bool
            Whether or not a drift occurred.

        Note: this method seems questionable. Do we need it?
        """
        return self.was_bucket_deleted

    def reset_change(self):
        self.was_bucket_deleted = False

    def detected_warning_zone(self):
        return False

    @property
    def variance(self):
        return self._variance / self.width

    @property
    def estimation(self):
        if self.width == 0:
            return 0
        return self.total / self.width

    # required by BaseDriftDetector
    @estimation.setter
    def estimation(self, value):
        pass

    def add_element(self, value):
        """ Add a new element to the sample window.

        Insert the value in the correct bucket & update the relevant
        statistics: sum of all values, window width, & total variance.

        Parameters
        ----------
        value: int or float (a numeric value)

        Notes
        -----
        The value parameter can be any numeric value relevant to the
        analysis of concept change. For the learners in this framework
        we are using either 0's or 1's, that are interpreted as
        follows:
        0: the learners prediction was wrong
        1: the learners prediction was correct

        This function should be called at every new sample analysed.
        """
        self.width += 1
        # Insert the new data.
        self.window[0].appendleft(Bucket(value, 0))
        if len(self.window) > self.max_n_buckets:
            self.max_n_buckets = len(self.window)

        if self.width > 1:
            incremental_variance = ((self.width - 1)
                                    * (value - self.total
                                       / (self.width - 1))
                                    * (value - self.total
                                       / (self.width - 1))
                                    / self.width)
        else:
            incremental_variance = 0

        self._variance += incremental_variance
        self.total += value
        self._compress_buckets()

    @staticmethod
    def bucket_size(bucket_list_index):
        return 2 ** bucket_list_index

    def delete_bucket(self):
        """ Delete a stale bucket from the window.

        Delete the oldest bucket and update the relevant statistics
        kept by ADWIN.

        Returns
        -------
        int
            The bucket size from the deleted bucket.
        """
        # Oldest bucket list.
        bucket_list = self.window[-1]
        n1 = self.bucket_size(len(self.window) - 1)
        self.width -= n1
        # Oldest bucket.
        bucket = bucket_list[-1]
        self.total -= bucket.total
        u1 = bucket.total / n1
        incremental_variance = (bucket.variance
                                + n1 * self.width
                                  * (u1 - self.total / self.width)
                                  * (u1 - self.total / self.width)
                                  / (n1 + self.width))
        self._variance -= incremental_variance
        bucket_list.pop()

        if bucket_list.size == 0:
            self.window.pop()

        return n1

    def _compress_buckets(self):
        """ Compress the bucket lists to a max length of MAX_BUCKETS.
        """
        # We might add a bucket list whilst iterating on the window.
        for i in range(len(self.window) + 1):
            bucket_list = self.window[i]
            if len(bucket_list) != self.MAX_BUCKETS + 1:
                return
            if i == len(self.window) - 1:
                self.window.append(deque())
            # Merge the two oldest buckets.
            b1 = bucket_list[-2]
            b2 = bucket_list[-1]
            next_bucket_list = self.window[i + 1]
            n1 = n2 = self.bucket_size(i)
            u1 = b1.total / n1
            u2 = b2.total / n2
            incremental_variance = n1 * n2 * (u1 - u2) * (u1 - u2) / (n1 + n2)
            new_variance = b1.variance + incremental_variance
            # Add the merged buckets to the next bucket list.
            next_bucket_list.appendleft(Bucket(b1.total + b2.total,
                                               new_variance))
            # Remove the merged buckets.
            bucket_list.pop()
            bucket_list.pop()

    def detected_change(self):
        """ Detect concept change in a drifting data stream.

        The ADWIN algorithm is described in Bifet and Gavaldà's
        'Learning from Time-Changing Data with Adaptive Windowing'.
        The general idea is to keep statistics from a window of
        variable size while detecting concept drift.

        This function runs every time the clock ticks & analyses the
        cutting point between each bucket to verify if there is a
        significant change in concept.

        Returns
        -------
        has_changed : bool
            Whether change was detected or not.

        Notes
        -----
        If change was detected, one should verify the new window size,
        by reading the width property.
        """
        has_changed = False
        should_exit = False
        was_bucket_deleted = False
        self.time += 1
        if ((self.time % self.clock == 0)
            and (self.width > self.min_window_longitude)):
            should_reducewidth = True
            while should_reducewidth:
                should_reducewidth = False
                should_exit = False
                n0 = u0 = v0 = 0
                n1 = self.width
                u1 = self.total
                v1 = self._variance
                n2 = u2 = 0
                i = len(self.window) - 1
                bucket_list = self.window[-1]
                while (not should_exit) and (bucket_list is not None):
                    bucket_list = self.window[i]
                    for k in range(bucket_list.size - 1):
                        n2 = self.bucket_size(i)
                        u2 = bucket_list.bucket_total[k]

                        if n0 > 0:
                            v0 += (bucket_list.bucket_variance[k]
                                   + n0
                                     * n2
                                     * (u0/n0 - u2/n2)
                                     * (u0/n0 - u2/n2)
                                     / (n0 + n2))
                        if n1 > 0:
                            v1 -= (bucket_list.bucket_variance[k]
                                   + n1
                                    * n2
                                    * (u1/n1 - u2/n2)
                                    * (u1/n1 - u2/n2)
                                    / (n1 + n2))

                        n0 += self.bucket_size(i)
                        n1 -= self.bucket_size(i)
                        u0 += bucket_list.bucket_total[k]
                        u1 -= bucket_list.bucket_total[k]

                        if (i == 0) and (k == bucket_list.size - 1):
                            should_exit = True
                            break

                        abs_value = u0 / n0 - u1 / n1
                        if ((n1 >= self.min_window_length)
                            and (n0 >= self.min_window_length)
                            and self._should_cut(n0, n1, u0, u1, v0, v1,
                                                 abs_value, self.delta)):
                            was_bucket_deleted = True
                            self.detect = self.time
                            if self.detect == 0:
                                self.detect = self.time
                            elif self.detect_twice == 0:
                                self.detect_twice = self.time

                            should_reducewidth = True
                            has_changed = True
                            if self.width > 0:
                                n0 -= self.delete_bucket_list()
                                should_exit = True
                                break
                    i -= 1
        self.total_width += self.width
        if has_changed:
            self.n_detections += 1
        self.in_concept_change = has_changed
        return has_changed

    def _should_cut(self, n0, n1, u0, u1, v0, v1, abs_value, delta):
        n = self.width
        dd = np.log(2 * np.log(n) / delta)
        v = self.variance
        m = ((1. / (n0 - self.min_window_length + 1))
             + (1. / (n1 - self.min_window_length + 1)))
        epsilon = np.sqrt(2 * m * v * dd) + 2. * dd * m / 3
        return np.absolute(abs_value) > epsilon
