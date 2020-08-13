import os
import pandas as pd
import numpy as np

from skmultiflow.data.base_stream import Stream
from skmultiflow.data.data_stream import check_data_consistency
from kafka import KafkaConsumer


class FileStream(Stream):
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
    >>> from skmultiflow.data.kafka_stream import FileStream
    >>> # Setup the stream
    >>> def parse_entry(line):
    >>>    return TODO complete
    >>> message_to_array_function = parse_entry
    >>> stream = FileStream(source, entry_to_array_function, ['localhost:9092'], 'topic-name', 'group-id')
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

    def __init__(self, source, entry_to_dictionary, n_targets=1, cat_features=None, allow_nan=False):
        super().__init__()
        self.source = source
        self.entry_to_dictionary = entry_to_dictionary
        self.n_targets = n_targets
        self.cat_features = cat_features
        self.cat_features_idx = [] if self.cat_features is None else self.cat_features
        self.allow_nan = allow_nan

        if self.cat_features_idx:
            if max(self.cat_features_idx) < self.n_features:
                self.n_cat_features = len(self.cat_features_idx)
            else:
                raise IndexError('Categorical feature index in {} exceeds n_features {}'
                                 .format(self.cat_features_idx, self.n_features))

        # Automatically infer target_idx if not passed in multi-output problems
        if self.n_targets > 1 and self.target_idx == -1:
            self.target_idx = -self.n_targets

        self.task_type = None
        self.n_classes = 0
        self.n_num_features = 0
        self.current_sample_x = None
        self.current_sample_y = None
        self.feature_names = None
        self.target_names = None
        self.target_values = None
        self.name = None


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
        return 'FileStream(filename={}, target_idx={}, n_targets={}, cat_features={})' \
            .format("'" + self.basename + "'", self.target_idx, self.n_targets, self.cat_features)
