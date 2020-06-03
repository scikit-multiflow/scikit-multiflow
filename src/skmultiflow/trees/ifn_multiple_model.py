import math
import pickle
import os
import numpy as np
from scipy import stats

from skmultiflow.trees.ifn.iolin import IncrementalOnlineNetwork
from skmultiflow.data import SEAGenerator


class MultipleModel(IncrementalOnlineNetwork):

    def __init__(self, classifier, path, number_of_classes=2, n_min=378, n_max=math.inf, alpha=0.99,
                 Pe=0.5, init_add_count=10, inc_add_count=50, max_add_count=100, red_add_count=75, min_add_count=1,
                 max_window=1000, data_stream_generator=SEAGenerator(random_state=1)):

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
        """ This function is an implementation of Multiple Model IOLIN algorithm as represented
            by Prof. Mark Last, et al. in "https://www.sciencedirect.com/science/article/abs/pii/S156849460800046X".
            This function updates a current model as long as the concept is stable.
            However, if a concept drift has been detected for the first time, the algorithm searches for
            the best model representing the current data from all previous networks.

        """

        self.window = self.meta_learning.calculate_Wint(self.Pe)
        self.classifier.window_size = self.window
        j = self.window
        add_count = self.init_add_count

        while j < self.n_max:

            X_batch, y_batch = self.data_stream_generator.next_sample(self.window)

            if not self.classifier.is_fitted:  # cold start
                self._induce_new_model(training_window_X=X_batch, training_window_y=y_batch)

            X_validation_samples, y_validation_samples = self.data_stream_generator.next_sample(add_count)
            j = j + add_count

            Etr = self.classifier.calculate_error_rate(X=X_batch,
                                                       y=y_batch)

            Eval = self.classifier.calculate_error_rate(X=X_validation_samples,
                                                        y=y_validation_samples)

            max_diff = self.meta_learning.get_max_diff(Etr, Eval, add_count)

            if abs(Eval - Etr) < max_diff:  # concept is stable
                self._update_current_network(training_window_X=X_batch,
                                             training_window_y=y_batch)
            else:
                unique, counts = np.unique(np.array(y_batch), return_counts=True)
                target_distribution_current = counts[0] / len(y_batch)
                # Entropy of target attribute on the current window
                E_current = stats.entropy([target_distribution_current, 1 - target_distribution_current], base=2)

                classifier_files_names = os.listdir(self.path)
                generated_classifiers = {}

                for classifier in classifier_files_names:
                    full_path = os.path.join(self.path, classifier)
                    generated_clf = pickle.load(open(full_path, "rb"))
                    # Entropy of target attribute on a former window
                    target_distribution_former = generated_clf.class_count[0][1] / len(y_batch)
                    E_former = stats.entropy([target_distribution_former, 1 - target_distribution_former], base=2)
                    generated_classifiers[classifier] = abs(E_current - E_former)

                # Choose network with min |E_current(T)–E_former(T)|
                chosen_classifier_name = min(generated_classifiers, key=generated_classifiers.get)
                full_path = os.path.join(self.path, chosen_classifier_name)
                chosen_classifier = pickle.load(open(full_path, "rb"))

                self.classifier = chosen_classifier
                self._new_split_process(training_window_X=X_batch)  # add new layer if possible

                # test the new chosen network on a new validation window
                X_validation_samples, y_validation_samples = self.data_stream_generator.next_sample(add_count)
                j = j + add_count

                # error rate of the new chosen classifier on the new window
                Etr = self.classifier.calculate_error_rate(X=X_batch,
                                                           y=y_batch)

                Eval = self.classifier.calculate_error_rate(X=X_validation_samples,
                                                            y=y_validation_samples)

                max_diff = self.meta_learning.get_max_diff(Etr, Eval, add_count)

                if abs(Eval - Etr) < max_diff:  # concept is stable
                    self._update_current_network(training_window_X=X_batch,
                                                 training_window_y=y_batch)

                # If concept drift is detected again with the chosen network create
                # completely new network using the Info-Fuzzy algorithm
                else:
                    self._induce_new_model(training_window_X=X_batch, training_window_y=y_batch)

            j = j + self.window

        full_path = os.path.join(self.path, str(self.counter - 1))
        last_model = pickle.load(open(full_path + ".pickle", "rb"))
        return last_model
