import os
import warnings
import re
from math import floor
from timeit import default_timer as timer
import numpy as np
import operator
from skmultiflow.evaluation.base_evaluator import StreamEvaluator
from skmultiflow.utils import constants
from statistics import mode
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats import ranksums

def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]


class EvaluateInfluential(StreamEvaluator):
    def __init__(self,
                 n_wait=200,
                 max_samples=100000,
                 n_time_windows=1,
                 batch_size=1,
                 pretrain_size=200,
                 n_intervals=2,
                 max_time=float("inf"),
                 metrics=None,
                 output_file=None,
                 show_plot=False,
                 restart_stream=True,
                 data_points_for_classification=False,
                 track_weight=False):

        super().__init__()
        self._method = 'Influential'
        self.n_wait = n_wait
        self.max_samples = max_samples
        self.n_time_windows = n_time_windows
        self.n_intervals = n_intervals
        self.pretrain_size = pretrain_size
        self.batch_size = batch_size
        self.max_time = max_time
        self.output_file = output_file
        self.show_plot = show_plot
        self.data_points_for_classification = data_points_for_classification
        self.window_size = (self.max_samples - self.pretrain_size) / self.n_time_windows
        self.distribution_table = []
        self.categorical_features = []
        self.numerical_features = []
        self.track_weight = track_weight
        self.weight_tracker = []

        if not self.data_points_for_classification:
            if metrics is None:
                self.metrics = [constants.ACCURACY, constants.KAPPA]

            else:
                if isinstance(metrics, list):
                    self.metrics = metrics
                else:
                    raise ValueError("Attribute 'metrics' must be 'None' or 'list', passed {}".format(type(metrics)))

        else:
            if metrics is None:
                self.metrics = [constants.DATA_POINTS]

            else:
                if isinstance(metrics, list):
                    self.metrics = metrics
                    self.metrics.append(constants.DATA_POINTS)
                else:
                    raise ValueError("Attribute 'metrics' must be 'None' or 'list', passed {}".format(type(metrics)))

        self.restart_stream = restart_stream
        self.n_sliding = n_wait

        warnings.filterwarnings("ignore", ".*invalid value encountered in true_divide.*")
        warnings.filterwarnings("ignore", ".*Passing 1d.*")

    def evaluate(self, stream, model, model_names=None):
        """ Evaluates a model or set of models on samples from a stream.

        Parameters
        ----------
        stream: Stream
            The stream from which to draw the samples.x

        model: skmultiflow.core.BaseStreamModel or sklearn.base.BaseEstimator or list
            The model or list of models to evaluate.

        model_names: list, optional (Default=None)
            A list with the names of the models.

        Returns
        -------
        StreamModel or list
            The trained model(s).

        """
        self._init_evaluation(model=model, stream=stream, model_names=model_names)

        if self._check_configuration():
            self._reset_globals()
            # Initialize metrics and outputs (plots, log files, ...)
            self._init_metrics()
            self._init_plot()
            self._init_file()

            self.model = self._train_and_test()
            if self.track_weight:
                self.weight_tracker = [self.stream.weight]

            if self.show_plot:
                self.visualizer.hold()

            return self.model

    def _train_and_test(self):
        """ Method to control the prequential evaluation.

        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifiers.

        Notes
        -----
        The classifier parameter should be an extension from the BaseClassifier. In
        the future, when BaseRegressor is created, it could be an extension from that
        class as well.

        """
        self._start_time = timer()
        self._end_time = timer()
        print('Influential Stream')
        print('Evaluating {} target(s).'.format(self.stream.n_targets))

        actual_max_samples = self.stream.n_remaining_samples()
        if actual_max_samples == -1 or actual_max_samples > self.max_samples:
            actual_max_samples = self.max_samples

        data_cache = []

        first_run = True
        if self.pretrain_size > 0:
            print('Pre-training on {} sample(s).'.format(self.pretrain_size))

            X, y = self.stream.next_sample(self.pretrain_size)

            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION:
                    # Training time computation
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y, classes=self.stream.target_values)
                    self.running_time_measurements[i].compute_training_time_end()
                elif self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y, classes=np.unique(self.stream.target_values))
                    self.running_time_measurements[i].compute_training_time_end()
                else:
                    self.running_time_measurements[i].compute_training_time_begin()
                    self.model[i].partial_fit(X=X, y=y)
                    self.running_time_measurements[i].compute_training_time_end()
                self.running_time_measurements[i].update_time_measurements(self.pretrain_size)
            self.global_sample_count += self.pretrain_size
            first_run = False

        update_count = 0
        window_count = 0
        interval_borders = []
        print('Evaluating...')
        while ((self.global_sample_count < actual_max_samples) & (self._end_time - self._start_time < self.max_time)
               & (self.stream.has_more_samples())):
            try:
                X, y = self.stream.next_sample(self.batch_size)
                if X is not None and y is not None:
                    # Test
                    prediction = [[] for _ in range(self.n_models)]
                    for i in range(self.n_models):
                        try:
                            # Testing time
                            self.running_time_measurements[i].compute_testing_time_begin()
                            prediction[i].extend(self.model[i].predict(X))
                            self.running_time_measurements[i].compute_testing_time_end()
                        except TypeError:
                            raise TypeError("Unexpected prediction value from {}"
                                            .format(type(self.model[i]).__name__))
                    self.global_sample_count += self.batch_size

                    for j in range(self.n_models):
                        for i in range(len(prediction[0])):
                            self.mean_eval_measurements[j].add_result(y[i], prediction[j][i])
                            self.current_eval_measurements[j].add_result(y[i], prediction[j][i])
                            self.stream.receive_feedback(y[i], prediction[j][i], X[i])
                            window_count += 1
                            data_instance = [y[i], prediction[j][i], X[i]]
                            data_cache.append(data_instance)
                            if window_count == self.window_size:
                                feature_data = [item[2] for item in data_cache]
                                interval_borders = self.create_intervals(feature_data)
                                self.init_table(feature_data, interval_borders)
                                self.count_update(data_cache, interval_borders, time_window=0)
                                data_cache = []
                            if window_count > self.window_size and window_count % self.window_size == 0:
                                feature_data = [item[2] for item in data_cache]
                                interval_borders = self.create_intervals(feature_data)
                                time_window = int(window_count / self.window_size - 1)
                                self.count_update(data_cache, interval_borders, time_window)
                                data_cache = []
                            if self.track_weight:
                                self.weight_tracker.append(self.stream.weight)
                    self._check_progress(actual_max_samples)

                    # Train
                    if first_run:
                        for i in range(self.n_models):
                            if self._task_type != constants.REGRESSION and \
                                    self._task_type != constants.MULTI_TARGET_REGRESSION:
                                # Accounts for the moment of training beginning
                                self.running_time_measurements[i].compute_training_time_begin()
                                self.model[i].partial_fit(X, y, self.stream.target_values)
                                # Accounts the ending of training
                                self.running_time_measurements[i].compute_training_time_end()
                            else:
                                self.running_time_measurements[i].compute_training_time_begin()
                                self.model[i].partial_fit(X, y)
                                self.running_time_measurements[i].compute_training_time_end()

                            # Update total running time
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)
                        first_run = False
                    else:
                        for i in range(self.n_models):
                            self.running_time_measurements[i].compute_training_time_begin()
                            self.model[i].partial_fit(X, y)
                            self.running_time_measurements[i].compute_training_time_end()
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)

                    if ((self.global_sample_count % self.n_wait) == 0 or
                            (self.global_sample_count >= self.max_samples) or
                            (self.global_sample_count / self.n_wait > update_count + 1)):
                        if prediction is not None:
                            self._update_metrics()
                        update_count += 1

                self._end_time = timer()

            except BaseException as exc:
                print(exc)
                if exc is KeyboardInterrupt:
                    self._update_metrics()
                break

        print("weights list: ", self.weight_tracker)
        # Flush file buffer, in case it contains data
        self._flush_file_buffer()
        self.evaluate_density()

        if len(set(self.metrics).difference({constants.DATA_POINTS})) > 0:
            self.evaluation_summary()
        else:
            print('Done')

        if self.restart_stream:
            self.stream.restart()

        return self.model

    def evaluate_density(self):
        # table = tn, fp, fn, tp
        for nfeature in range(self.stream.n_features):
            # dist table is organized like this: [tn, fp, fn, tp],[tn, fp, fn, tp],[tn, fp, fn, tp],[tn, fp, fn, tp]
            # next step will create the following: [tn, tn, tn, tn],[fp,fp,fp,fp] etc
            t0 = preprocessing.normalize(list(zip(*self.distribution_table[0][nfeature])))
            t1 = preprocessing.normalize(list(zip(*self.distribution_table[1][nfeature])))

            t0 = list(zip(*self.distribution_table[0][nfeature]))
            t1 = list(zip(*self.distribution_table[1][nfeature]))
            t0t1_list = []
            # this for loop adds the density of the same piece from t0 and t1 together
            for i in range(4):
                t0t1_list.append(list(zip(t1[i], t0[i])))
            temp_list = []
            diff_list = []

            # this subtracts the density in t0 from the density of t1
            for item in t0t1_list:
                for x, y in item:
                    if abs(x) > 10 and abs(y) > 10:
                        temp_list.append(x - y)
                    else:
                        temp_list.append(None)
                if None in temp_list:
                    temp_list = [None]
                diff_list.append(temp_list)
                temp_list = []

            # diff_list[2] is false negative and [3] is true positive
            if diff_list[2][0] is not None and diff_list[3][0] is not None:
                diff_list[2] = preprocessing.normalize([diff_list[2]])
                diff_list[3] = preprocessing.normalize([diff_list[3]])
                print(ranksums(diff_list[2], diff_list[3]))
            else:
                print("too little samples")
            # [0] is true negative and [1] is false positive
            if diff_list[0][0] is not None and diff_list[1][0] is not None:
                diff_list[0] = preprocessing.normalize([diff_list[0]])
                diff_list[1] = preprocessing.normalize([diff_list[1]])
                print(ranksums(diff_list[0], diff_list[1]))
            else:
                print("too little samples")

    def create_intervals(self, feature_data):
        values_per_feature = list(zip(*feature_data))

        percentile_steps = 100 / self.n_intervals
        percentile_values = list(np.arange(percentile_steps, 101, percentile_steps))
        interval_borders = list(map(lambda feature: np.percentile(feature, percentile_values).tolist(),
                                    values_per_feature))
        categorical_values_per_feature = self.get_categorical_features(values_per_feature)
        idx = 0
        for categorical in self.categorical_features:
            interval_borders[categorical] = categorical_values_per_feature[idx]
            idx += 1
        # print("initial intervals percentiles: ", interval_borders)
        return interval_borders

    def init_table(self, feature_data, interval_borders):
        values_per_feature = list(zip(*feature_data))
        # print("values per feature: ", values_per_feature)
        unique_values_per_feature = list(map(set, values_per_feature))
        unique_values_per_feature = list(map(list, unique_values_per_feature))
        # print("unique_values_per_feature ", unique_values_per_feature)
        values_per_categorical_feature = self.get_categorical_features(values_per_feature)
        self.distribution_table = [[[[0] * 4 for _ in range(self.n_intervals)] for _ in range(self.stream.n_features)]
                                   for _ in range(self.n_time_windows)]
        # remove intervals for categorical values:
        for time in range(self.n_time_windows):
            for categorical in self.categorical_features:
                self.distribution_table[time][categorical] = [[0] * 4 for _ in
                                                              range(len(unique_values_per_feature[categorical]))]
        # print("initialized distribution table: ", self.distribution_table)

    def get_categorical_features(self, values_per_feature):
        mode_per_feature = list(map(mode, values_per_feature))
        self.numerical_features = []
        self.categorical_features = []

        unique_values_per_feature = list(map(set, values_per_feature))

        for x in range(self.stream.n_features):
            values_per_feature[x] = remove_values_from_list(values_per_feature[x], mode_per_feature[x])

        unique_values_per_feature_without_mode = list(map(set, values_per_feature))
        idx = 0
        unique_value_limit = 0.005 * self.window_size
        unique_value_limit = 0.005 * self.window_size

        for x in unique_values_per_feature_without_mode:
            if len(x) < unique_value_limit:
                self.categorical_features.append(idx)
            else:
                self.numerical_features.append(idx)
            idx += 1

        categories = [unique_values_per_feature[i] for i in self.categorical_features]
        categories = list(map(list, categories))
        # print("Categorical features are: ", self.categorical_features)
        # print("categories are: ", categories)
        return categories

    def count_update(self, data_cache, interval_borders, time_window):
        # data_instance = [y[i], prediction[j][i], X[i]]
        # order of distribution table = tn, fp, fn, tp
        for data_instance in data_cache:
            true_label = data_instance[0]
            prediction = data_instance[1]
            if true_label == 0 and prediction == 0:
                # true negative
                self.count_update_helper(data_instance, interval_borders, time_window, cf=0)
            elif true_label == 0 and prediction == 1:
                # false positive
                self.count_update_helper(data_instance, interval_borders, time_window, cf=1)
            elif true_label == 1 and prediction == 0:
                # false negative
                self.count_update_helper(data_instance, interval_borders, time_window, cf=2)
            elif true_label == 1 and prediction == 1:
                # true positive
                self.count_update_helper(data_instance, interval_borders, time_window, cf=3)

    def count_update_helper(self, data_instance, interval_borders, time_window, cf):
        for feature in range(self.stream.n_features):
            if feature in self.categorical_features:
                feature_index = 0
                for interval in range(len(interval_borders[feature])):
                    if data_instance[2][feature] == interval_borders[feature][interval]:
                        self.distribution_table[time_window][feature][feature_index][cf] += 1
                    feature_index += 1
                continue
            check = False
            for interval in range(self.n_intervals):
                if data_instance[2][feature] <= interval_borders[feature][interval]:
                    # print(data_instance[2][feature], " < ", interval_borders[feature][interval])
                    check = True
                    self.distribution_table[time_window][feature][interval][cf] += 1
                    break
            if not check:
                self.distribution_table[time_window][feature][self.n_intervals - 1][cf] += 1

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially fit all the models on the given data.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.

        y: Array-like
            An array-like containing the classification labels / target values for all samples in X.

        classes: list
            Stores all the classes that may be encountered during the classification task. Not used for regressors.

        sample_weight: Array-like
            Samples weight. If not provided, uniform weights are assumed.

        Returns
        -------
        EvaluatePrequential
            self

        """
        if self.model is not None:
            for i in range(self.n_models):
                if self._task_type == constants.CLASSIFICATION or \
                        self._task_type == constants.MULTI_TARGET_CLASSIFICATION:
                    self.model[i].partial_fit(X=X, y=y, classes=classes, sample_weight=sample_weight)
                else:
                    self.model[i].partial_fit(X=X, y=y, sample_weight=sample_weight)
            return self
        else:
            return self

    def predict(self, X):
        """ Predicts with the estimator(s) being evaluated.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.

        Returns
        -------
        list of numpy.ndarray
            Model(s) predictions

        """
        predictions = None
        if self.model is not None:
            predictions = []
            for i in range(self.n_models):
                predictions.append(self.model[i].predict(X))

        return predictions

    def get_info(self):
        info = self.__repr__()
        if self.output_file is not None:
            _, filename = os.path.split(self.output_file)
            info = re.sub(r"output_file=(.\S+),", "output_file='{}',".format(filename), info)

        return info
