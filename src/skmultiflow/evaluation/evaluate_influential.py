import os
import warnings
import re
from timeit import default_timer as timer
import numpy as np
from skmultiflow.evaluation.base_evaluator import StreamEvaluator
from skmultiflow.utils import constants
from statistics import mode
from scipy.stats import ranksums
import pandas as pd
import matplotlib.pyplot as plt


class EvaluateInfluential(StreamEvaluator):
    def __init__(self,
                 n_wait=200,
                 max_samples=100000,
                 n_time_windows=1,
                 batch_size=1,
                 batch_size_update=1,
                 pretrain_size=200,
                 n_intervals=2,
                 max_time=float("inf"),
                 metrics=None,
                 output_file=None,
                 show_plot=False,
                 restart_stream=True,
                 data_points_for_classification=False,
                 weight_output=True,
                 weight_plot=False
                 ):

        super().__init__()
        self._method = 'Influential'
        self.n_wait = n_wait
        self.max_samples = max_samples
        self.n_time_windows = n_time_windows
        self.n_intervals = n_intervals
        self.pretrain_size = pretrain_size
        self.batch_size = batch_size
        self.batch_size_update = batch_size_update
        self.max_time = max_time
        self.output_file = output_file
        self.show_plot = show_plot
        self.data_points_for_classification = data_points_for_classification
        self.window_size = (self.max_samples - self.pretrain_size) / self.n_time_windows
        self.distribution_table = []
        self.categorical_features = []
        self.numerical_features = []
        self.table_influence_on_positive = []
        self.table_influence_on_negative = []
        self.weight_output = weight_output
        self.weight_plot = weight_plot
        self.accuracy = []

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

            if self.show_plot:
                self.visualizer.hold()

            return self.model

    def _train_and_test(self):
        """ Method to control the evaluation.

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
        update_cache = []
        first_run = True
        if self.pretrain_size > 0:
            print('Pre-training on {} sample(s).'.format(self.pretrain_size))
            for sample in range(self.pretrain_size):
                X, y, used_stream = self.stream.next_sample(self.batch_size)
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
                    self.running_time_measurements[i].update_time_measurements(self.batch_size)
            self.global_sample_count += self.pretrain_size
            first_run = False

        update_count = 0
        window_count = 0
        print('Evaluating...')

        while ((self.global_sample_count < actual_max_samples) & (self._end_time - self._start_time < self.max_time)
               & (self.stream.has_more_samples())):
            try:
                X, y, used_stream = self.stream.next_sample(self.batch_size)
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
                            window_count += 1
                            data_instance = [y[i], prediction[j][i], X[i], used_stream]
                            data_cache.append(data_instance)
                            update_cache.append(data_instance)
                            if window_count % self.batch_size_update == 0:
                                true_label = [item[0] for item in update_cache]
                                prediction = [item[1] for item in update_cache]
                                features = [item[2] for item in update_cache]
                                stream = [item[3] for item in update_cache]
                                self.stream.receive_feedback(true_label, prediction, features, stream)
                                update_cache = []
                            if window_count == self.window_size:
                                feature_data = [item[2] for item in data_cache]
                                interval_borders = self.create_intervals(feature_data)
                                self.init_table(feature_data)
                                self.count_update(data_cache, interval_borders, time_window=0)
                                data_cache = []
                            if window_count > self.window_size and window_count % self.window_size == 0:
                                feature_data = [item[2] for item in data_cache]
                                interval_borders = self.create_intervals(feature_data)
                                time_window = int(window_count / self.window_size - 1)
                                self.count_update(data_cache, interval_borders, time_window)
                                data_cache = []
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

        # Flush file buffer, in case it contains data
        self._flush_file_buffer()

        if self.weight_output:
            print("weights: ", self.stream.weight)

        if self.weight_plot:
            self.stream.plot_weight()
        self.calculate_density()
        self.accuracy = self._data_buffer.get_data(metric_id=constants.ACCURACY, data_id=constants.MEAN)

        if len(set(self.metrics).difference({constants.DATA_POINTS})) > 0:
            self.evaluation_summary()
        else:
            print('Done')

        if self.restart_stream:
            self.stream.restart()

        return self.model

    def calculate_density(self):
        # table = tn, fp, fn, tp
        # create list that has two values (difference in density in time0 and time1 of TP and FN,
        #                               and difference in density in time0 and time1 of TN and FP per interval
        TN = 0
        FP = 1
        FN = 2
        TP = 3
        density = [[[[0] * 2 for _ in range(self.n_intervals)] for _ in range(self.stream.n_features)]
                   for _ in range(self.n_time_windows - 1)]
        subset_TP = [[[] for _ in range(self.stream.n_features)] for _ in range(self.n_time_windows - 1)]
        subset_FN = [[[] for _ in range(self.stream.n_features)] for _ in range(self.n_time_windows - 1)]
        subset_TN = [[[] for _ in range(self.stream.n_features)] for _ in range(self.n_time_windows - 1)]
        subset_FP = [[[] for _ in range(self.stream.n_features)] for _ in range(self.n_time_windows - 1)]
        for timeperiod1 in range(self.n_time_windows - 1):
            timeperiod2 = timeperiod1 + 1
            for feature in range(self.stream.n_features):
                # dist table is organized like this: [tn, fp, fn, tp],[tn, fp, fn, tp],[tn, fp, fn, tp],[tn, fp, fn, tp]
                # next step will create the following: [tn, tn, tn, tn],[fp,fp,fp,fp] etc (if there are 4 intervals)
                # print("distribution table: ", self.distribution_table)

                t0 = list(zip(*self.distribution_table[timeperiod1][feature]))
                # create table with density differences of TP and FN (0), and TN and FP (1), per interval
                for interval in range(self.n_intervals):
                    # calculate density of positive instances = instances of TP + FN / window size
                    density_0 = (self.distribution_table[timeperiod1][feature][interval][TP] +
                                 self.distribution_table[timeperiod1][feature][interval][FN]) / self.window_size
                    density_1 = (self.distribution_table[timeperiod2][feature][interval][TP] +
                                 self.distribution_table[timeperiod2][feature][interval][FN]) / self.window_size
                    # density differences is density1 - density0
                    density_difference = density_1 - density_0

                    # fill in density difference in list
                    density[timeperiod1][feature][interval][0] = density_difference

                    # calculate density of negative instances
                    density_0 = (self.distribution_table[timeperiod1][feature][interval][TN] +
                                 self.distribution_table[timeperiod1][feature][interval][FP]) / self.window_size
                    density_1 = (self.distribution_table[timeperiod2][feature][interval][TN] +
                                 self.distribution_table[timeperiod2][feature][interval][FP]) / self.window_size
                    # density differences is density1 in density0
                    density_difference = density_1 - density_0
                    density[timeperiod1][feature][interval][1] = density_difference

                for interval in range(self.n_intervals):
                    # add the amount of instances per interval that is belonging to subset
                    # so if feature0, interval 0 has 6 TP instances, and the calculated difference
                    # in density of feature0,
                    # interval 0 is 0.07, you will extend the subset with [0.07,0.07,0.07,0.07,0.07,0.07]
                    subset_TP[timeperiod1][feature].extend(
                        [density[timeperiod1][feature][interval][0]] * t0[TP][interval])
                    subset_FN[timeperiod1][feature].extend(
                        [density[timeperiod1][feature][interval][0]] * t0[FN][interval])
                    subset_FP[timeperiod1][feature].extend(
                        [density[timeperiod1][feature][interval][1]] * t0[FP][interval])
                    subset_TN[timeperiod1][feature].extend(
                        [density[timeperiod1][feature][interval][1]] * t0[TN][interval])
        self.test_density(subset_TN, subset_FP, subset_FN, subset_TP)

    def test_density(self, subset_TN, subset_FP, subset_FN, subset_TP):
        for timeperiod in range(self.n_time_windows - 1):
            for feature in range(self.stream.n_features):
                TP = subset_TP[timeperiod][feature]
                FN = subset_FN[timeperiod][feature]
                TN = subset_TN[timeperiod][feature]
                FP = subset_FP[timeperiod][feature]
                # positive instances:
                mean_subset_TP = 0
                mean_subset_FN = 0
                if len(subset_TP[timeperiod][feature]) > 0:
                    mean_subset_TP = sum(TP) / len(TP)
                if len(subset_FN[timeperiod][feature]) > 0:
                    mean_subset_FN = sum(FN) / len(FN)
                if len(TP) > 10 and len(FN) > 10:
                    test = ranksums(TP, FN)
                    result = test.pvalue
                else:
                    result = None
                self.table_influence_on_positive.append([feature, timeperiod, len(TP),
                                                         mean_subset_TP, len(FN),
                                                         mean_subset_FN, abs(mean_subset_TP - mean_subset_FN), result])

                # negative instances
                mean_subset_TN = 0
                mean_subset_FP = 0
                if len(TN) > 0:
                    mean_subset_TN = sum(TN) / len(TN)
                if len(FP) > 0:
                    mean_subset_FP = sum(FP) / len(FP)
                if len(TN) > 10 and len(FP) > 10:
                    test = ranksums(TN, FP)
                    result = test.pvalue
                else:
                    result = None
                self.table_influence_on_negative.append([feature, timeperiod, len(TN),
                                                         mean_subset_TN, len(FP),
                                                         mean_subset_FP, abs(mean_subset_TN - mean_subset_FP), result])

    def create_intervals(self, feature_data):
        values_per_feature = list(zip(*feature_data))

        percentile_steps = 100 / self.n_intervals
        percentile_values = list(np.arange(percentile_steps, 101, percentile_steps))
        if percentile_values[-1] > 100:
            percentile_values[-1] = 100
        interval_borders = list(map(lambda feature: np.percentile(feature, percentile_values).tolist(),
                                    values_per_feature))
        categorical_values_per_feature = self.get_categorical_features(values_per_feature)
        idx = 0
        for categorical in self.categorical_features:
            interval_borders[categorical] = categorical_values_per_feature[idx]
            idx += 1
        return interval_borders

    def init_table(self, feature_data):
        values_per_feature = list(zip(*feature_data))
        unique_values_per_feature = list(map(set, values_per_feature))
        unique_values_per_feature = list(map(list, unique_values_per_feature))

        self.distribution_table = [[[[0] * 4 for _ in range(self.n_intervals)] for _ in range(self.stream.n_features)]
                                   for _ in range(self.n_time_windows)]
        # remove intervals for categorical values:
        for time in range(self.n_time_windows):
            for categorical in self.categorical_features:
                self.distribution_table[time][categorical] = [[0] * 4 for _ in
                                                              range(len(unique_values_per_feature[categorical]))]

    def remove_values_from_list(self, the_list, val):
        return [value for value in the_list if value != val]

    def get_categorical_features(self, values_per_feature):
        mode_per_feature = list(map(mode, values_per_feature))
        self.numerical_features = []
        self.categorical_features = []

        unique_values_per_feature = list(map(set, values_per_feature))

        for x in range(self.stream.n_features):
            values_per_feature[x] = self.remove_values_from_list(values_per_feature[x], mode_per_feature[x])

        unique_values_per_feature_without_mode = list(map(set, values_per_feature))
        idx = 0
        unique_value_limit = 0.005 * self.window_size

        for x in unique_values_per_feature_without_mode:
            if len(x) < unique_value_limit:
                self.categorical_features.append(idx)
            else:
                self.numerical_features.append(idx)
            idx += 1

        categories = [unique_values_per_feature[i] for i in self.categorical_features]
        categories = list(map(list, categories))
        return categories

    def count_update(self, data_cache, interval_borders, time_window):
        # data_instance = [y[i], prediction[j][i], X[i]]
        # order of distribution table = tn, fp, fn, tp
        TN = 0
        FP = 1
        FN = 2
        TP = 3
        for data_instance in data_cache:
            true_label = data_instance[0]
            prediction = data_instance[1]
            if true_label == 0 and prediction == 0:
                # true negative
                self.count_update_helper(data_instance, interval_borders, time_window, cf=TN)
            elif true_label == 0 and prediction == 1:
                # false positive
                self.count_update_helper(data_instance, interval_borders, time_window, cf=FP)
            elif true_label == 1 and prediction == 0:
                # false negative
                self.count_update_helper(data_instance, interval_borders, time_window, cf=FN)
            elif true_label == 1 and prediction == 1:
                # true positive
                self.count_update_helper(data_instance, interval_borders, time_window, cf=TP)

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
