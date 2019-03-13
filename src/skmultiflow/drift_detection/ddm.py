import numpy as np
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class DDM(BaseDriftDetector):
    """ DDM method for concept drift detection
    
    Parameters
    ----------
    min_num_instances: int (default=30)
        The minimum required number of analyzed samples so change can be 
        detected. This is used to avoid false detections during the early 
        moments of the detector, when the weight of one sample is important.

    warning_level: float (default=2.0)
        Warning Level

    out_control_level: float (default=3.0)
        Out-control Level

    Notes
    -----
    DDM (Drift Detection Method) [1]_ is a concept change detection method
    based on the PAC learning model premise, that the learner's error rate
    will decrease as the number of analysed samples increase, as long as the
    data distribution is stationary.

    If the algorithm detects an increase in the error rate, that surpasses
    a calculated threshold, either change is detected or the algorithm will
    warn the user that change may occur in the near future, which is called
    the warning zone.

    The detection threshold is calculated in function of two statistics,
    obtained when `(pi + si)` is minimum:

    * `pmin`: The minimum recorded error rate.
    * `smin`: The minimum recorded standard deviation.

    At instant `i`, the detection algorithm uses:

    * `pi`: The error rate at instant i.
    * `si`: The standard deviation at instant i.

    The conditions for entering the warning zone and detecting change are
    as follows:

    * if `pi + si >= pmin + 2 * smin` -> Warning zone
    * if `pi + si >= pmin + 3 * smin` -> Change detected

    References
    ----------
    .. [1] João Gama, Pedro Medas, Gladys Castillo, Pedro Pereira Rodrigues: Learning
       with Drift Detection. SBIA 2004: 286-295

    Examples
    --------
    >>> # Imports
    >>> import numpy as np
    >>> from skmultiflow.drift_detection import DDM
    >>> ddm = DDM()
    >>> # Simulating a data stream as a normal distribution of 1's and 0's
    >>> data_stream = np.random.randint(2, size=2000)
    >>> # Changing the data concept from index 999 to 1500, simulating an 
    >>> # increase in error rate
    >>> for i in range(999, 1500):
    ...     data_stream[i] = 0
    >>> # Adding stream elements to DDM and verifying if drift occurred
    >>> for i in range(2000):
    ...     ddm.add_element(data_stream[i])
    ...     if ddm.detected_warning_zone():
    ...         print('Warning zone has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
    ...     if ddm.detected_change():
    ...         print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))

    """

    def __init__(self, min_num_instances=30, warning_level=2.0, out_control_level=3.0):
        super().__init__()
        self._init_min_num_instances = min_num_instances
        self._init_warning_level = warning_level
        self._init_out_control = out_control_level
        self.sample_count = None
        self.miss_prob = None
        self.miss_std = None
        self.miss_prob_sd_min = None
        self.miss_prob_min = None
        self.miss_sd_min = None
        self.min_instances = None
        self.warning_level = None
        self.out_control_level = None
        self.reset()

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        super().reset()
        self.sample_count = 1
        self.miss_prob = 1.0
        self.miss_std = 0.0
        self.miss_prob_sd_min = float("inf")
        self.miss_prob_min = float("inf")
        self.miss_sd_min = float("inf")
        self.min_instances = self._init_min_num_instances
        self.warning_level = self._init_warning_level
        self.out_control_level = self._init_out_control

    def add_element(self, prediction):
        """ Add a new element to the statistics
        
        Parameters
        ----------
        prediction: int (either 0 or 1)
            This parameter indicates whether the last sample analyzed was
            correctly classified or not. 1 indicates an error (miss-classification).
        
        Notes
        -----
        After calling this method, to verify if change was detected or if  
        the learner is in the warning zone, one should call the super method 
        detected_change, which returns True if concept drift was detected and
        False otherwise.
        
        """
        if self.in_concept_change:
            self.reset()

        self.miss_prob = self.miss_prob + (prediction - self.miss_prob) / float(self.sample_count)
        self.miss_std = np.sqrt(self.miss_prob * (1 - self.miss_prob) / float(self.sample_count))
        self.sample_count += 1

        self.estimation = self.miss_prob
        self.in_concept_change = False
        self.in_warning_zone = False
        self.delay = 0

        if self.sample_count < self.min_instances:
            return

        if self.miss_prob + self.miss_std <= self.miss_prob_sd_min:
            self.miss_prob_min = self.miss_prob
            self.miss_sd_min = self.miss_std
            self.miss_prob_sd_min = self.miss_prob + self.miss_std

        if self.miss_prob + self.miss_std > self.miss_prob_min + self.out_control_level * self.miss_sd_min:
            self.in_concept_change = True

        elif self.miss_prob + self.miss_std > self.miss_prob_min + self.warning_level * self.miss_sd_min:
            self.in_warning_zone = True

        else:
            self.in_warning_zone = False

    def get_info(self):
        """ Collect information about the concept drift detector.

        Returns
        -------
        string
            Configuration for the concept drift detector.
        """
        description = type(self).__name__ + ': '
        description += 'min_num_instances: {} - '.format(self.min_instances)
        description += 'warning_level: {} - '.format(self.warning_level)
        description += 'out_control_level: {} - '.format(self.out_control_level)
        return description
