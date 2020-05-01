import math
import pickle

import pandas as pd

from skmultiflow.data import SEAGenerator
from skmultiflow.trees import IfnClassifier

from skmultiflow.trees.ifn.olin import OnlineNetwork


class BasicIncremental(OnlineNetwork):

    def __init__(self, classifier: IfnClassifier, path, number_of_classes=2, n_min=378, n_max=math.inf, alpha=0.99,
                 Pe=0.5, init_add_count=10, inc_add_count=50, max_add_count=100, red_add_count=75, min_add_count=1,
                 max_window=1000, data_stream_generator=SEAGenerator()):

        super().__init__(classifier, path, number_of_classes, n_min, n_max, alpha, Pe, init_add_count, inc_add_count,
                         max_add_count, red_add_count, min_add_count, max_window, data_stream_generator)

    def generate(self):

        self.window = self.meta_learning.calculate_Wint(self.Pe)
        i = self.n_min - self.window
        j = self.window
        add_count = self.init_add_count
        X_batch = []
        y_batch = []

        while j < self.n_max:

            while i < j:
                X, y = self.data_stream_generator.next_sample()
                X_batch.append(X[0])
                y_batch.append(y[0])
                i = i + 1

            if self.classifier.is_fitted: # the network already fitted at least one time before

                k = j + add_count
                X_validation_samples = []
                y_validation_samples = []

                while j < k:
                    X_validation_sample, y_validation_sample = self.data_stream_generator.next_sample()
                    X_validation_samples.append(X_validation_sample[0])
                    y_validation_samples.append(y_validation_sample[0])
                    j = j + 1

                self._incremental_IN(X_batch, y_batch, X_validation_samples, y_validation_samples, add_count)
                i = j - self.window

            else:  # cold start
                X_batch_df = pd.DataFrame(X_batch)
                self.classifier.fit(X_batch_df, y_batch)
                j = j + self.window
                # save the model
                path = self.path + "/" + str(self.counter)
                pickle.dump(self.classifier, open(path, "wb"))
                self.counter = self.counter + 1

            j = j + self.window
            X_batch.clear()
            y_batch.clear()

    def _incremental_IN(self,
                        training_window_X,
                        training_window_y,
                        validation_window_X,
                        validation_window_y,
                        add_count):

        Etr = self.classifier.calculate_error_rate(X=training_window_X,
                                                   y=training_window_y)

        Eval = self.classifier.calculate_error_rate(X=validation_window_X,
                                                    y=validation_window_y)

        max_diff = self.meta_learning.get_max_diff(Etr=Etr,
                                                   Eval=Eval,
                                                   add_count=add_count)

        if abs(Eval - Etr) < max_diff:  # concept is stable
            self._update_current_network(training_window_X=training_window_X,
                                         training_window_y=training_window_y)
        else:  # concept drift detected
            self._induce_new_model(training_window_X=training_window_X,
                                   training_window_y=training_window_y)
