import numpy as np

from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class ADWIN(BaseDriftDetector):
    """Adaptive Windowing method for concept drift detection.

    Parameters
    ----------
    delta : float (default=0.002)
        The delta parameter for the ADWIN algorithm.

    Notes
    -----
    ADWIN [1]_ (ADaptive WINdowing) is an adaptive sliding window
    algorithm for detecting change, and keeping updated statistics
    about a data stream. ADWIN allows algorithms not adapted for
    drifting data, to be resistant to this phenomenon.

    The general idea is to keep statistics from a window of variable
    size while detecting concept drift.

    The algorithm will decide the size of the window by cutting the
    statistics' window at different points and analysing the average of
    some statistic over these two windows. If the absolute value of the
    difference between the two averages surpasses a pre-defined
    threshold, change is detected at that point and all data before
    that time is discarded.

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

    MAX_BUCKETS = 5

    def __init__(self, delta=.002):
        """ADWIN Init.

        The sliding window is stored in `bucket_rows` as a list of
        `Row`s, each one keeping a list of buckets of the same size.

        """
        super().__init__()
        # default values affected by init_bucket()
        self.delta = delta
        self.last_bucket_row = 0
        self.bucket_rows = None
        self.total = 0
        self._variance = 0
        self.width = 0
        self.n_buckets = 0

        self.__init__buckets()

        # other default values
        self.min_window_longitude = 10

        self.time = 0
        self.width_t = 0

        self.detect = 0
        self.n_detections = 0
        self.detect_twice = 0
        self.clock = 32

        self.was_bucket_deleted = False
        self.max_n_buckets = 0
        self.min_window_length = 5
        super().reset()

    def reset(self):
        """Reset detectors.

        Resets statistics and adwin's window.

        Returns
        -------
        ADWIN
            self

        """
        self.__init__(delta=self.delta)

    def get_change(self):
        """Get drift.

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

    def __init__buckets(self):
        """Initialize the buckets List and statistics.

        Set all statistics to 0 and create a new bucket List.

        """
        self.bucket_rows = List()
        self.last_bucket_row = 0
        self.total = 0
        self._variance = 0
        self.width = 0
        self.n_buckets = 0

    def add_element(self, value):
        """Add a new element to the sample window.

        Apart from adding the element value to the window, by inserting
        it in the correct bucket, it will also update the relevant
        statistics, in this case the total sum of all values, the
        window width, and the total variance.

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
        self._insert_element_bucket(0, value, self.bucket_rows.first)

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

    def _insert_element_bucket(self, variance, value, row):
        row.insert_bucket(value, variance)
        self.n_buckets += 1

        if self.n_buckets > self.max_n_buckets:
            self.max_n_buckets = self.n_buckets

    @staticmethod
    def bucket_size(row):
        return 2 ** row

    def delete_row(self):
        """Delete a Row from the bucket list.

        Deletes the last Row and updates relevant statistics kept by
        ADWIN.

        Returns
        -------
        int
            The bucket size from the updated bucket.

        """
        row = self.bucket_rows.last
        n1 = self.bucket_size(self.last_bucket_row)
        self.width -= n1
        self.total -= row.buckettotal[0]
        u1 = row.buckettotal[0] / n1
        incremental_variance = (row.bucket_variance[0]
                                + n1
                                  * self.width
                                  * (u1 - self.total / self.width)
                                  * (u1 - self.total / self.width)
                                  / (n1 + self.width))
        self._variance -= incremental_variance
        row.remove_bucket()
        self.n_buckets -= 1

        if row.size == 0:
            self.bucket_rows.remove_from_tail()
            self.last_bucket_row -= 1

        return n1

    def _compress_buckets(self):
        row = self.bucket_rows.first
        i = 0
        while row is not None:
            if row.size != self.MAX_BUCKETS + 1:
                break
            next_row = row.next
            if next_row is None:
                self.bucket_rows.add_to_tail()
                next_row = row.next
                self.last_bucket_row += 1
            n1 = n2 = self.bucket_size(i)
            u1 = row.buckettotal[0]/n1
            u2 = row.buckettotal[1]/n2
            incremental_variance = n1 * n2 * (u1 - u2) * (u1 - u2) / (n1 + n2)
            next_row.insert_bucket(row.buckettotal[0] + row.buckettotal[1],
                                   row.bucket_variance[1]
                                   + incremental_variance)
            self.n_buckets += 1
            row.compress_bucket_row(2)

            row = row.next
            i += 1

    def detected_change(self):
        """Detect concept change in a drifting data stream.

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
                row = self.bucket_rows.last
                i = self.last_bucket_row

                while (not should_exit) and (row is not None):
                    for k in range(row.size - 1):
                        n2 = self.bucket_size(i)
                        u2 = row.buckettotal[k]

                        if n0 > 0:
                            v0 += (row.bucket_variance[k]
                                   + n0
                                     * n2
                                     * (u0/n0 - u2/n2)
                                     * (u0/n0 - u2/n2)
                                     / (n0 + n2))
                        if n1 > 0:
                            v1 -= (row.bucket_variance[k]
                                   + n1
                                    * n2
                                    * (u1/n1 - u2/n2)
                                    * (u1/n1 - u2/n2)
                                    / (n1 + n2))

                        n0 += self.bucket_size(i)
                        n1 -= self.bucket_size(i)
                        u0 += row.buckettotal[k]
                        u1 -= row.buckettotal[k]

                        if (i == 0) and (k == row.size - 1):
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
                                n0 -= self.delete_row()
                                should_exit = True
                                break

                    row = row.previous
                    i -= 1
        self.width_t += self.width
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


class List(object):
    """A doubly-linked list object for ADWIN algorithm.

    Used for storing ADWIN's bucket list. Is composed of Row objects.
    Acts as a doubly-linked list, where each element points to its
    predecessor and successor.

    """

    def __init__(self):
        super().__init__()
        self.size = None
        self.first = None
        self.last = None
        self.reset()
        self.add_to_head()

    def reset(self):
        self.size = 0
        self.first = None
        self.last = None

    def add_to_head(self):
        self.first = Row(self.first, None)
        if self.last is None:
            self.last = self.first

    def remove_from_head(self):
        self.first = self.first.next
        if self.first is not None:
            self.first.previous = None
        else:
            self.last = None
        self.size -= 1

    def add_to_tail(self):
        self.last = Row(None, self.last)
        if self.first is None:
            self.first = self.last
        self.size += 1

    def remove_from_tail(self):
        self.last = self.last.previous
        if self.last is not None:
            self.last.next = None
        else:
            self.first = None
        self.size -= 1


class Row(object):
    """Row of buckets of the same size.

    The Row object, alongside the List object, are the two main data
    structures used for storing the relevant statistics for the ADWIN
    algorithm for change detection.

    Parameters
    ----------
    next_item: Row object
        Reference to the next Row in the List
    previous_item: Row object
        Reference to the previous Row in the List

    """
    def __init__(self, next=None, previous=None):
        super().__init__()
        self.next = next
        self.previous = previous
        if next is not None:
            next.previous = self
        if previous is not None:
            previous.next = self
        self.size = None
        self.max_buckets = ADWIN.MAX_BUCKETS
        self.buckettotal = np.zeros(self.max_buckets+1, dtype=float)
        self.bucket_variance = np.zeros(self.max_buckets+1, dtype=float)
        self.reset()

    def reset(self):
        """Reset the algorithm's statistics and window.

        Returns
        -------
        ADWIN
            self.

        """
        self.size = 0
        for i in range(ADWIN.MAX_BUCKETS + 1):
            self._clear_bucket(i)

        return self

    def _clear_bucket(self, index):
        self.buckettotal[index] = 0
        self.bucket_variance[index] = 0

    def insert_bucket(self, value, variance):
        new_item = self.size
        self.size += 1
        self.buckettotal[new_item] = value
        self.bucket_variance[new_item] = variance

    def remove_bucket(self):
        self.compress_bucket_row(1)

    def compress_bucket_row(self, num_deleted=1):
        for i in range(num_deleted, ADWIN.MAX_BUCKETS + 1):
            self.buckettotal[i-num_deleted] = self.buckettotal[i]
            self.bucket_variance[i-num_deleted] = self.bucket_variance[i]

        for i in range(1, num_deleted+1):
            self._clear_bucket(ADWIN.MAX_BUCKETS - i + 1)

        self.size -= num_deleted
