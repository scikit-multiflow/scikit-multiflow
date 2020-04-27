import math
import pickle

import numpy as np
import pandas as pd
from skmultiflow.trees.IFN.IOLIN import OnlineNetwork
from skmultiflow.trees.IFN import IfnClassifier
from skmultiflow.data import SEAGenerator


class OnlineNetworkRegenerative(OnlineNetwork):

    def __init__(self, classifier: IfnClassifier, path, number_of_classes=2, n_min=378, n_max=math.inf, alpha=0.99,
                 Pe=0.5, init_add_count=10, inc_add_count=50, max_add_count=100, red_add_count=75, min_add_count=1,
                 max_window=1000, data_stream_generator=SEAGenerator()):

        """
        Parameters
        ----------
        classifier :
        path : String
            A path to save the model.
        number_of_classes : int
            The number of classes in the target.
        n_min : int
            The number of the first example to be classified by the system.
        n_max : int
            The number of the last example to be classified by the system.
            (if unspecified, the system will run indefinitely).
        alpha : float
            Significance level
        Pe : float
            Maximum allowable prediction error of the model.
        init_add_count : int
            The number of new examples to be classified by the first model.
        inc_add_count : int
            Amount (percentage) to increase the number of examples between model re-constructions.
        max_add_count : int
            Maximum number of examples between model re-constructions.
        red_add_count : int
            Amount (percentage) to reduce the number of examples between model reconstructions.
        min_add_count : int
            Minimum number of examples between model re-constructions.
        max_window : int
            Maximum number of examples in a training window.
        data_stream_generator : stream generator
            Stream generator for the stream data
        """

        super().__init__(classifier, path, number_of_classes, n_min, n_max, alpha, Pe, init_add_count, inc_add_count,
                         max_add_count, red_add_count, min_add_count, max_window, data_stream_generator)

    def generate(self):
        """ This function is an implementation to the regenerative algorithm as represented
            by Prof. Mark Last, et al in "https://content.iospress.com/articles/intelligent-data-analysis/ida00083".

            This function obtain an IFN model for every window arriving in the stream,
            and validate the prediction on the next window, which represent the validation examples.
            The size of the window depends on the stability of the information, if the data is stable,
            the size of the window increase, otherwise (a concept drift detect) the size of the window
            re-calculate.

            After each iteration the IFN model is saved to a file in the path given by the user.

        """

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

            X_batch_df = pd.DataFrame(X_batch)
            self.classifier.fit(X_batch_df, y_batch)
            Etr = self.classifier.calculate_error_rate(X_batch, y_batch)

            k = j + add_count
            X_validation_samples = []
            y_validation_samples = []

            while j < k:
                X_validation, y_validation = self.data_stream_generator.next_sample()
                X_validation_samples.append(X_validation[0])
                y_validation_samples.append(y_validation[0])
                j = j + 1

            j = k

            Eval = self.classifier.calculate_error_rate(X_validation_samples, y_validation_samples)
            max_diff = self.meta_learning.get_max_diff(Etr, Eval, add_count)

            if abs(Eval - Etr) < max_diff:  # concept is stable
                add_count = min(add_count * (1 + (self.inc_add_count / 100)), self.max_add_count)
                self.window = min(self.window + add_count, self.max_window)
                self.meta_learning.window = self.window
                i = j - self.window

            else:  # concept drift detected
                unique, counts = np.unique(np.array(y_batch), return_counts=True)
                target_distribution = counts[0] / len(y_batch)
                NI = len(self.classifier.network.root_node.first_layer.nodes)
                self.window = self.meta_learning.calculate_new_window(NI, target_distribution, Etr)
                i = j - self.window
                add_count = max(add_count * (1 - (self.red_add_count / 100)), self.min_add_count)

            path = self.path + "/" + str(self.counter)
            pickle.dump(self.classifier, open(path, "wb"))
            self.counter = self.counter + 1
            X_batch.clear()
            y_batch.clear()
