import numpy as np
import random
from skmultiflow.data.base_stream import Stream
from skmultiflow.utils import check_random_state
from skmultiflow.data import AGRAWALGenerator
from skmultiflow.data.random_rbf_generator import RandomRBFGenerator
from skmultiflow.data.random_rbf_generator_drift import RandomRBFGeneratorDrift


class InfluentialStream(Stream):

    def __init__(self, streams=None,
                 random_state=None,
                 weight=None,
                 self_fulfilling=1.1,
                 self_defeating=0.9,
                 count=1):
        super(InfluentialStream, self).__init__()

        if streams is None:
            streams = [RandomRBFGenerator(model_random_state=99, sample_random_state=50, n_classes=2,
                                          n_features=2, n_centroids=25),
                       RandomRBFGeneratorDrift(model_random_state=112, sample_random_state=50, n_classes=2,
                                               n_features=2, n_centroids=25, change_speed=0.9,
                                               num_drift_centroids=5)]
        self.streams = streams
        self.n_samples = streams[0].n_samples
        self.n_targets = streams[0].n_targets
        self.n_features = streams[0].n_features
        self.n_num_features = streams[0].n_num_features
        self.n_cat_features = streams[0].n_cat_features
        self.n_classes = streams[0].n_classes
        self.cat_features_idx = streams[0].cat_features_idx
        self.feature_names = streams[0].feature_names
        self.target_names = streams[0].target_names
        self.target_values = streams[0].target_values
        self.n_targets = streams[0].n_targets
        self.name = streams[0].name

        self.weight = weight
        self.weight_tracker = []
        self.last_stream = None
        self.self_fulfilling = self_fulfilling
        self.self_defeating = self_defeating
        self.count = count
        self.cache = []

        self.random_state = random_state
        self._random_state = None  # This is the actual random_state object used internally

        self._prepare_for_use()
        self.set_weight()

    def _prepare_for_use(self):
        self._random_state = check_random_state(self.random_state)

    def set_weight(self):
        if self.weight is None:
            counter = len(self.streams)
            start = [1] * counter
            self.weight = [1] * counter
        self.weight_tracker = [start]

    def check_weight(self):
        print("the current weights are: ", self.weight)

    def n_remaining_samples(self):
        """ Returns the estimated number of remaining samples.

        Returns
        -------
        int
            Remaining number of samples. -1 if infinite (e.g. generator)
        """
        n_samples = -1
        for stream in self.streams:
            n_samples += stream.n_remaining_samples()
        if n_samples < 0:
            n_samples = -1
        return n_samples

    def has_more_samples(self):
        """ Checks if stream has more samples.

        Returns
        -------
        Boolean
            True if stream has more samples.
        """
        for stream in self.streams:
            if not stream.has_more_samples():
                return False
        return True

    def is_restartable(self):
        """ Determine if the stream is restartable.

         Returns
         -------
         Boolean
            True if stream is restartable.
         """
        for stream in self.streams:
            if not stream.is_restartable():
                return False
        return True

    def next_sample(self, batch_size=1):
        """ Returns next sample from the stream.

        Parameters
        ----------
        batch_size: int (optional, default=1)
            The number of samples to return.

        Returns
        -------
        tuple or tuple list
            Return a tuple with the features matrix
            for the batch_size samples that were requested.

        """
        self.current_sample_x = np.zeros((batch_size, self.n_features))
        self.current_sample_y = np.zeros((batch_size, self.n_targets))

        for j in range(batch_size):
            self.sample_idx += 1
            n_streams = list(range(len(self.streams)))
            probability = random.choices(n_streams, self.weight)
            used_stream = probability[0]
            for stream in range(len(self.weight)):
                if stream == used_stream:
                    X, y = self.streams[stream].next_sample()
                    self.last_stream = stream

            self.current_sample_x[j, :] = X
            self.current_sample_y[j, :] = y

        return self.current_sample_x, self.current_sample_y.flatten()

    def receive_feedback(self, y_true, y_pred, x_features):
        """
        If the true label is given in this function and
        if the cache is empty or the instance matches the first item in the list
        then apply the self_fulfilling weight to the stream if correctly classified
        or apply the self_defeating weight to the stream if incorrectly classified
        If the cache is not empty, this means we received feedback on the first instance in the cache.
        After giving feedback, this instance can be removed.
        If after this instance, an instance with three items (y_true, y_pred, and x_features) is the first
        in the list, we can also give that instance feedback.

        If a true label is given, but the cache is not empty and the instance does not match with the
        first instance, add the instance to the end of the cache.

        If no true label is given, then add the instance in the end of the cache.
        """
        if y_true is not None:
            if len(self.cache) == 0 or (y_pred == self.cache[0][0] and x_features == self.cache[0][1]):
                self.receive_feedback_update(y_true, y_pred)
                if len(self.cache) != 0:
                    self.cache.remove(self.cache[0])
                    while len(self.cache[0]) == 3:
                        self.receive_feedback_update(y_true, y_pred)
                        self.cache.remove(self.cache[0])
            else:
                wait_for_feedback = [y_pred, x_features, y_true]
                self.cache.append(wait_for_feedback)
        else:
            no_label = [y_pred, x_features]
            self.cache.append(no_label)

    def receive_feedback_update(self, y_true, y_pred):
        if y_true == y_pred:
            self.weight[self.last_stream] = self.weight[self.last_stream] * self.self_fulfilling
        else:
            self.weight[self.last_stream] = self.weight[self.last_stream] * self.self_defeating
        self.weight_tracker.append(self.weight.copy())

    def restart(self):
        self._random_state = check_random_state(self.random_state)
        self.sample_idx = 0
        for stream in self.streams:
            stream.restart()
