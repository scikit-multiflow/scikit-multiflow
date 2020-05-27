import math
import pickle
import os
import numpy as np
from skmultiflow.data import SEAGenerator
from skmultiflow.trees.ifn.meta_learning import MetaLearning


class OnlineNetworkRegenerative():

    def __init__(self, classifier, path, number_of_classes=2, n_min=378, n_max=math.inf, alpha=0.99,
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

        self.classifier = classifier
        self.path = path
        self.number_of_classes = number_of_classes
        self.n_min = n_min
        self.n_max = n_max
        self.alpha = alpha
        self.Pe = Pe
        self.init_add_count = init_add_count
        self.inc_add_count = inc_add_count
        self.max_add_count = max_add_count
        self.red_add_count = red_add_count
        self.min_add_count = min_add_count
        self.max_window = max_window
        self.window = None
        self.meta_learning = MetaLearning(alpha, number_of_classes)
        self.data_stream_generator = data_stream_generator
        self.data_stream_generator.prepare_for_use()
        self.counter = 1

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
        self.classifier.window_size = self.window
        j = self.window
        add_count = self.init_add_count

        while j < self.n_max:

            X_batch, y_batch = self.data_stream_generator.next_sample(self.window)

            if len(X_batch) == 0:
                break

            self.classifier.partial_fit(X_batch, y_batch)
            Etr = self.classifier.calculate_error_rate(X_batch, y_batch)

            X_validation_samples, y_validation_samples = self.data_stream_generator.next_sample(int(add_count))
            j = j + add_count

            Eval = self.classifier.calculate_error_rate(X_validation_samples, y_validation_samples)
            max_diff = self.meta_learning.get_max_diff(Etr, Eval, add_count)

            if abs(Eval - Etr) < max_diff:  # concept is stable
                add_count = min(add_count * (1 + (self.inc_add_count / 100)), self.max_add_count)
                self.window = min(self.window + add_count, self.max_window)
                self.meta_learning.window = self.window
                j = j + self.window

            else:  # concept drift detected
                unique, counts = np.unique(np.array(y_batch), return_counts=True)
                target_distribution = counts[0] / len(y_batch)
                NI = len(self.classifier.network.root_node.first_layer.nodes)
                self.window = self.meta_learning.calculate_new_window(NI, target_distribution, Etr)
                j = j + self.window
                add_count = max(add_count * (1 - (self.red_add_count / 100)), self.min_add_count)

            full_path = os.path.join(self.path, str(self.counter))
            pickle.dump(self.classifier, open(full_path + ".pickle", "wb"))
            self.counter = self.counter + 1

        full_path = os.path.join(self.path, str(self.counter - 1))
        last_model = pickle.load(open(full_path + ".pickle", "rb"))
        return last_model
