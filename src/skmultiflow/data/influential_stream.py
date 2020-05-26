import numpy as np
import random
from skmultiflow.data.base_stream import Stream
from skmultiflow.utils import check_random_state
from skmultiflow.data import AGRAWALGenerator


class InfluentialStream(Stream):

    def __init__(self, streams=None,
                 random_state=None,
                 weight=None,
                 self_fulfilling=1.1,
                 self_defeating=0.9,
                 count=1):
        super(InfluentialStream, self).__init__()

        if streams is None:
            streams = [AGRAWALGenerator(random_state=110),
                       AGRAWALGenerator(random_state=120),
                       AGRAWALGenerator(random_state=130)]
        for i in range(len(streams)):
            self.n_samples = streams[i].n_samples
            self.n_targets = streams[i].n_targets
            self.n_features = streams[i].n_features
            self.n_num_features = streams[i].n_num_features
            self.n_cat_features = streams[i].n_cat_features
            self.n_classes = streams[i].n_classes
            self.cat_features_idx = streams[i].cat_features_idx
            self.feature_names = streams[i].feature_names
            self.target_names = streams[i].target_names
            self.target_values = streams[i].target_values
            self.n_targets = streams[i].n_targets
            self.name = streams[i].name
        self.weight = weight
        self.last_stream = None
        self.self_fulfilling = self_fulfilling
        self.self_defeating = self_defeating
        self.count = count

        self.random_state = random_state
        self._random_state = None  # This is the actual random_state object used internally
        self.streams = streams

        self._prepare_for_use()
        self.set_weight()

    def _prepare_for_use(self):
        self._random_state = check_random_state(self.random_state)

    def set_weight(self):
        if self.weight is None:
            counter = len(self.streams)
            self.weight = [1] * counter

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
            num_streams = list(range(len(self.streams)))
            probability = random.choices(num_streams, self.weight)
            used_stream = probability[0]
            for stream in range(len(self.weight)):
                if stream == used_stream:
                    X, y = self.streams[stream].next_sample()
                    self.last_stream = stream

            self.current_sample_x[j, :] = X
            self.current_sample_y[j, :] = y

        return self.current_sample_x, self.current_sample_y.flatten()

    def receive_feedback(self, y_true, y_pred):
        # TODO: add features (x), or add index of samples, change
        """This checks which stream was used last, and checks whether the
        prediction of the last sample was correct.

        If the sample was correctly classified, the weight of the last
        used stream can be increased by multiplying the weight
         by a set value,
        if it is incorrectly classified, it is decreased by multiplying
         the weight with a set value."""

        for i in range(len(self.streams)):
            if self.last_stream == i:
                if y_true == y_pred:
                    self.weight[i] = self.weight[i] * self.self_fulfilling
                else:
                    self.weight[i] = self.weight[i] * self.self_defeating

    def restart(self):
        self._random_state = check_random_state(self.random_state)
        self.sample_idx = 0
        for stream in self.streams:
            stream.restart()
