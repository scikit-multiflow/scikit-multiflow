from collections import deque
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
       " Learning from time-changing data with adaptive windowing."
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

    MAX_BUCKETS = 5

    def __init__(self, delta=.002):
        """ ADWIN Init.

        The sliding window is stored in `window` as a deque of
        BucketList, each one keeping a list of buckets of the same
        size.
        """
        super().__init__()
        self.delta = delta
        self.window = deque([BucketList()])
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

        This function should be used at every new sample analysed.
        """
        self.width += 1
        self._insert_element_bucket(0, value, self.window[0])

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

    def _insert_element_bucket(self, variance, value, bucket_list):
        bucket_list.insert_bucket(value, variance)

        if len(self.window) > self.max_n_buckets:
            self.max_n_buckets = len(self.window)

    @staticmethod
    def bucket_size(bucket_list_index):
        return 2 ** bucket_list_index

    def delete_bucket_list(self):
        """ Delete a BucketList from the bucket list.

        Delete the last BucketList and update the relevant statistics
        kept by ADWIN.

        Returns
        -------
        int
            The bucket size from the updated bucket.
        """
        bucket_list = self.window[-1]
        n1 = self.bucket_size(len(self.window))
        self.width -= n1
        self.total -= bucket_list.bucket_total[0]
        u1 = bucket_list.bucket_total[0] / n1
        incremental_variance = (bucket_list.bucket_variance[0]
                                + n1
                                  * self.width
                                  * (u1 - self.total / self.width)
                                  * (u1 - self.total / self.width)
                                  / (n1 + self.width))
        self._variance -= incremental_variance
        bucket_list.remove_bucket()

        if bucket_list.size == 0:
            self.window.pop()

        return n1

    def _compress_buckets(self):
        i = 0
        while i >= 0:
            bucket_list = self.window[i]
            if bucket_list.size != self.MAX_BUCKETS + 1:
                break
            if i == len(self.window) - 1:
                self.window.append(BucketList())
            next_bucket_list = self.window[i + 1]
            n1 = n2 = self.bucket_size(i)
            u1 = bucket_list.bucket_total[0] / n1
            u2 = bucket_list.bucket_total[1] / n2
            incremental_variance = n1 * n2 * (u1 - u2) * (u1 - u2) / (n1 + n2)
            next_bucket_list.insert_bucket(bucket_list.bucket_total[0] + bucket_list.bucket_total[1],
                                   bucket_list.bucket_variance[1]
                                   + incremental_variance)
            bucket_list.compress_bucket_bucket_list(2)
            i += 1

    def detected_change(self):
        """ Detect concept change in a drifting data stream.

        The ADWIN algorithm is described in Bifet and Gavaldà's
        'Learning from Time-Changing Data with Adaptive Windowing'.
        The general idea is to keep statistics from a window of
        variable size while detecting concept drift.

        This function is responsible for analysing different cutting
        points in the sliding window, to verify if there is a
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


class BucketList(object):
    """ List of buckets of the same size.

    A deque of BucketList is the main data structure used to store
    the relevant statistics for the ADWIN algorithm for change
    detection.
    """

    def __init__(self):
        super().__init__()
        self.size = 0
        self.bucket_total = np.zeros(ADWIN.MAX_BUCKETS + 1, dtype=float)
        self.bucket_variance = np.zeros(ADWIN.MAX_BUCKETS + 1, dtype=float)

    def _clear_bucket(self, index):
        self.bucket_total[index] = 0
        self.bucket_variance[index] = 0

    def insert_bucket(self, value, variance):
        self.bucket_total[self.size] = value
        self.bucket_variance[self.size] = variance
        self.size += 1

    def remove_bucket(self):
        self.compress_bucket_bucket_list(1)

    def compress_bucket_bucket_list(self, num_deleted=1):
        for i in range(num_deleted, ADWIN.MAX_BUCKETS + 1):
            self.bucket_total[i - num_deleted] = self.bucket_total[i]
            self.bucket_variance[i - num_deleted] = self.bucket_variance[i]

        self.bucket_total[-num_deleted:] = 0
        self.bucket_variance[-num_deleted:] = 0
        self.size -= num_deleted
