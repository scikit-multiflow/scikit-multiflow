import os
import pandas as pd
import numpy as np

from skmultiflow.data.base_stream import Stream
from skmultiflow.data.data_stream import check_data_consistency
from kafka import KafkaConsumer


class KafkaStream(Stream):
    """ Creates a stream from a Kafka source.

    Parameters
    ----------
    message_to_array_function: function.
        Function to parse Kafka message and turn it into a tuple with (X, y) values

    bootstrap_servers: array of strings
        Kafka bootstrap servers to which the consumer shall connect to

    topic: string
        Kafka topic to which we are willing to connect

    group_id: string, optional
        Kafka group id

    n_targets: int, optional (default=1)
        The number of targets.

    cat_features: list, optional (default=None)
        A list of indices corresponding to the location of categorical features.

    allow_nan: bool, optional (default=False)
        If True, allows NaN values in the data. Otherwise, an error is raised.

    Notes
    -----
    The stream object provides upon request a number of samples, in a way such that old samples
    cannot be accessed at a later time. This is done to correctly simulate the stream context.

    Examples
    --------
    >>> # Imports
    >>> from skmultiflow.data.kafka_stream import KafkaStream
    >>> # Setup the stream
    >>> def parse_message(msg):
    >>>    return np.array(msg.value.X, msg.value.y)
    >>> message_to_array_function = parse_message
    >>> stream = KafkaStream(message_to_array_function, ['localhost:9092'], 'topic-name', 'group-id')
    >>> # Retrieving one sample
    >>> stream.next_sample()
    (array([[0.080429, 8.397187, 7.074928]]), array([0]))
    >>> # Retrieving 10 samples
    >>> stream.next_sample(10)
    (array([[1.42074 , 7.504724, 6.764101],
        [0.960543, 5.168416, 8.298959],
        [3.367279, 6.797711, 4.857875],
        [9.265933, 8.548432, 2.460325],
        [7.295862, 2.373183, 3.427656],
        [9.289001, 3.280215, 3.154171],
        [0.279599, 7.340643, 3.729721],
        [4.387696, 1.97443 , 6.447183],
        [2.933823, 7.150514, 2.566901],
        [4.303049, 1.471813, 9.078151]]),
        array([0, 0, 1, 1, 1, 1, 0, 0, 1, 0]))
    >>> stream.n_remaining_samples()
    -1
    >>> stream.has_more_samples()
    True

    """
    _CLASSIFICATION = 'classification'
    _REGRESSION = 'regression'

    def __init__(self, message_to_array_function, bootstrap_servers, topic, group_id=None, n_targets=1, cat_features=None, allow_nan=False):
        super().__init__()
        self.message_to_array_function = message_to_array_function
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.n_targets = n_targets
        self.cat_features = cat_features
        self.cat_features_idx = [] if self.cat_features is None else self.cat_features
        self.allow_nan = allow_nan

        self.task_type = None
        self.n_classes = 0

        # Automatically infer target_idx if not passed in multi-output problems
        if self.n_targets > 1 and self.target_idx == -1:
            self.target_idx = -self.n_targets

        self._prepare_for_use()

    @property
    def n_targets(self):
        """
         Get the number of targets.

        Returns
        -------
        int:
            The number of targets.
        """
        return self._n_targets

    @n_targets.setter
    def n_targets(self, n_targets):
        """
        Sets the number of targets.

        Parameters
        ----------
        n_targets: int
        """

        self._n_targets = n_targets

    @property
    def cat_features_idx(self):
        """
        Get the list of the categorical features index.

        Returns
        -------
        list:
            List of categorical features index.

        """
        return self._cat_features_idx

    @cat_features_idx.setter
    def cat_features_idx(self, cat_features_idx):
        """
        Sets the list of the categorical features index.

        Parameters
        ----------
        cat_features_idx:
            List of categorical features index.
        """

        self._cat_features_idx = cat_features_idx


    def _prepare_for_use(self):
        self._connect()

    def _connect(self):
        if(self.group_id == None):
            self.consumer = KafkaConsumer(self.topic, self.group_id, bootstrap_servers=self.bootstrap_servers,
                                          auto_offset_reset='earliest', enable_auto_commit=True,
                                          auto_commit_interval_ms=1000)
        else:
            self.consumer = KafkaConsumer(self.topic, bootstrap_servers=self.bootstrap_servers,
                                          auto_offset_reset='earliest', enable_auto_commit=True,
                                          auto_commit_interval_ms=1000)


    def _process_data(self, raw_data):
        """ Reads the data provided by the user and separates the features and targets.
        """
        # TODO re-implement to fit new case
        #check_data_consistency(raw_data, self.allow_nan)

        self.n_samples = len(raw_data)
        X = np.array([instance['X'] for instance in raw_data])
        y = np.array([instance['y'] for instance in raw_data])

        _, self.n_features = self.X.shape
        if self.cat_features_idx:
            if max(self.cat_features_idx) < self.n_features:
                self.n_cat_features = len(self.cat_features_idx)
            else:
                raise IndexError('Categorical feature index in {} exceeds n_features {}'
                                 .format(self.cat_features_idx, self.n_features))
        self.n_num_features = self.n_features - self.n_cat_features

        if np.issubdtype(self.y.dtype, np.integer):
            self.task_type = self._CLASSIFICATION
            self.n_classes = len(np.unique(self.y))
        else:
            self.task_type = self._REGRESSION
        self.target_values = self.get_target_values()

        return X, y

    def next_sample(self, batch_size=1):
        """ Returns next sample from the stream.

        If there is enough instances to supply at least batch_size samples, those
        are returned. If there aren't, tuples of (None, None) are returned.

        Parameters
        ----------
        batch_size: int (optional, default=1)
            The number of instances to return.

        Returns
        -------
        tuple or tuple list
            Returns the next batch_size instances.
            For general purposes the return can be treated as a numpy.ndarray.

        """
        # The consumer shall consume and buffer messages, which are then consumed
        messages = []
        for count in range(0, batch_size):
            message = self.consumer.get_message(block=False, timeout=self.IterTimeout, get_partition_info=True )
            array = self.message_to_array_function(message)
            messages.append(array)

        X, y = self._process_data(messages)

        self.sample_idx += batch_size
        try:
            self.current_sample_x = X
            self.current_sample_y = y
            if self.n_targets < 2:
                self.current_sample_y = self.current_sample_y.flatten()

        return self.current_sample_x, self.current_sample_y

    def has_more_samples(self):
        """ Checks if stream has more samples.

        Returns
        -------
        Boolean
            True if stream has more samples.

        """
        return True

    def n_remaining_samples(self):
        """ Returns the estimated number of remaining samples.

        Returns
        -------
        int
            Remaining number of samples.

        """
        return -1

    def get_data_info(self):
        if self.task_type == self._CLASSIFICATION:
            return "{} - {} target(s), {} classes".format(self.basename, self.n_targets,
                                                          self.n_classes)
        elif self.task_type == self._REGRESSION:
            return "{} - {} target(s)".format(self.basename, self.n_targets)

    def get_target_values(self):
        if self.task_type == 'classification':
            if self.n_targets == 1:
                return np.unique(self.y).tolist()
            else:
                return [np.unique(self.y[:, i]).tolist() for i in range(self.n_targets)]
        elif self.task_type == self._REGRESSION:
            return [float] * self.n_targets

    def get_info(self):
        return 'KafkaStream(filename={}, target_idx={}, n_targets={}, cat_features={})' \
            .format("'" + self.basename + "'", self.target_idx, self.n_targets, self.cat_features)
