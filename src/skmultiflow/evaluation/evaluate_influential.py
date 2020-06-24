import os
import warnings
import re
from math import floor
from timeit import default_timer as timer
import numpy as np
import operator
from skmultiflow.evaluation.base_evaluator import StreamEvaluator
from skmultiflow.utils import constants


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
                 data_points_for_classification=False):

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
        self.window_size = self.max_samples / self.n_time_windows - self.pretrain_size
        self.distribution_table = []
        self.categorical_features = []
        self.numerical_features = []

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
            The stream from which to draw the samples.

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

        print("distribution table: ", self.distribution_table)
        # self.distribution_table[0][0][0][0] = 5
        # print("distribution table: ", self.distribution_table)

        # Flush file buffer, in case it contains data
        self._flush_file_buffer()

        if len(set(self.metrics).difference({constants.DATA_POINTS})) > 0:
            self.evaluation_summary()
        else:
            print('Done')

        if self.restart_stream:
            self.stream.restart()

        return self.model

    def create_intervals(self, feature_data):
        percentile_steps = 100 / self.n_intervals
        percentile_values = list(np.arange(percentile_steps, 101, percentile_steps))
        interval_borders = list(map(lambda feature: np.percentile(feature, percentile_values).tolist(),
                                    zip(*feature_data)))
        print("initial intervals percentiles: ", interval_borders)
        return interval_borders

    def init_table(self, feature_data, interval_borders):
        # get unique values per feature
        unique_values_per_feature = list(map(set, zip(*feature_data)))
        idx = 0
        for x in unique_values_per_feature:
            if len(x) < 25:
                self.categorical_features.append(idx)
            else:
                self.numerical_features.append(idx)
            idx += 1
        # self.distribution_table = [[[[0] * 4] * self.n_intervals] * self.stream.n_features] * self.n_time_windows
        self.distribution_table = [[[[0] * 4 for _ in range(self.n_intervals)] for _ in range(self.stream.n_features)]
                                   for _ in range(self.n_time_windows)]
        # remove intervals for categorical values:
        for time in range(self.n_time_windows):
            for categorical in self.categorical_features:
                self.distribution_table[time][categorical] = [0, 0, 0, 0]
                interval_borders[categorical] = [0, 0, 0, 0]
        print("initialized distribution table: ", self.distribution_table)

    def count_update(self, data_cache, interval_borders, time_window):
        # data_instance = [y[i], prediction[j][i], X[i]]
        # order of distribution table = tn, fp, fn, tp
        for data_instance in data_cache:
            true_label = data_instance[0]
            prediction = data_instance[1]
            if true_label == 0 and prediction == 0:
                # true negative
                cf = 0
                self.count_update_helper(data_instance, interval_borders, time_window, cf)
            elif true_label == 0 and prediction == 1:
                # false positive
                cf = 1
                self.count_update_helper(data_instance, interval_borders, time_window, cf)
            elif true_label == 1 and prediction == 0:
                # false negative
                cf = 2
                self.count_update_helper(data_instance, interval_borders, time_window, cf)
            elif true_label == 1 and prediction == 1:
                # true positive
                cf = 3
                self.count_update_helper(data_instance, interval_borders, time_window, cf)

    def count_update_helper(self, data_instance, interval_borders, time_window, cf):
        for feature in range(self.stream.n_features):
            if feature in self.categorical_features:
                self.distribution_table[time_window][feature][cf] += 1
                # print("categorical value: ", self.distribution_table[time_window][feature][cf])
                continue
            check = False
            for interval in range(self.n_intervals):
                if data_instance[2][feature] < interval_borders[feature][interval]:
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
