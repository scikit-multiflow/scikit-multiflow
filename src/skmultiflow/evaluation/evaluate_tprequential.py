import os
import warnings
import re
from timeit import default_timer as timer

from numpy import unique

from skmultiflow.evaluation.nbase_evaluator import NStreamEvaluator
from skmultiflow.utils import constants


class EvaluateTPrequential(NStreamEvaluator):
    """ The prequential evaluation method or interleaved test-then-train method.

    An alternative to the traditional holdout evaluation, inherited from
    batch setting problems.

    The prequential evaluation is designed specifically for stream settings,
    in the sense that each sample serves two purposes, and that samples are
    analysed sequentially, in order of arrival, and become immediately
    inaccessible.

    This method consists of using each sample to test the model, which means
    to make a predictions, and then the same sample is used to train the model
    (partial fit). This way the model is always tested on samples that it
    hasn't seen yet.

    Parameters
    ----------
    max_timestamps: int (Default: 24)
        The maximum number of timestamps to consider as window to process during the evaluation.

    batch_size: int (Default: 1)
        The number of samples to pass at a time to the model(s).

    pretrain_timestamps: int (Default: 3)
        The number of timestamps to use to train the model before starting the evaluation. Used to enforce a 'warm' start.

    max_time: float (Default: float("inf"))
        The maximum duration of the simulation (in seconds).

    metrics: list, optional (Default: ['accuracy', 'kappa'])
        | The list of metrics to track during the evaluation. Also defines the metrics that will be displayed in plots
          and/or logged into the output file. Valid options are
        | **Classification**
        | 'accuracy'
        | 'kappa'
        | 'kappa_t'
        | 'kappa_m'
        | 'true_vs_predicted'
        | 'precision'
        | 'recall'
        | 'f1'
        | 'gmean'
        | **Multi-target Classification**
        | 'hamming_score'
        | 'hamming_loss'
        | 'exact_match'
        | 'j_index'
        | **Regression**
        | 'mean_square_error'
        | 'mean_absolute_error'
        | 'true_vs_predicted'
        | **Multi-target Regression**
        | 'average_mean_squared_error'
        | 'average_mean_absolute_error'
        | 'average_root_mean_square_error'
        | **Experimental** (no plot generated)
        | 'running_time'
        | 'model_size'

    output_file: string, optional (Default: None)
        File name to save the summary of the evaluation.

    show_plot: bool (Default: False)
        If True, a plot will show the progress of the evaluation. Warning: Plotting can slow down the evaluation
        process.

    restart_stream: bool, optional (default: True)
        If True, the stream is restarted once the evaluation is complete.

    Notes
    -----
    1. This evaluator can process a single learner to track its performance; or multiple learners  at a time, to
       compare different models on the same stream.

    2. The metric 'true_vs_predicted' is intended to be informative only. It corresponds to evaluations at a specific
       moment which might not represent the actual learner performance across all instances.

    3. The metrics `running_time` and `model_size ` are not plotted when the `show_plot` option is set. Only their
       current value is displayed at the bottom of the figure. However, their values over the evaluation are written
       into the resulting csv file if the `output_file` option is set.

    Examples
    --------
    >>> # The first example demonstrates how to evaluate one model
    >>> from skmultiflow.data import SEAGenerator
    >>> from skmultiflow.trees import HoeffdingTreeClassifier
    >>> from skmultiflow.evaluation import EvaluatePrequential
    >>>
    >>> # Set the stream
    >>> stream = SEAGenerator(random_state=1)
    >>>
    >>> # Set the model
    >>> ht = HoeffdingTreeClassifier()
    >>>
    >>> # Set the evaluator
    >>>
    >>> evaluator = EvaluatePrequential(max_samples=24,
    >>>                                 max_time=1000,
    >>>                                 show_plot=True,
    >>>                                 metrics=['accuracy', 'kappa'])
    >>>
    >>> # Run evaluation
    >>> evaluator.evaluate(stream=stream, model=ht, model_names=['HT'])

    >>> # The second example demonstrates how to compare two models
    >>> from skmultiflow.data import SEAGenerator
    >>> from skmultiflow.trees import HoeffdingTreeClassifier
    >>> from skmultiflow.bayes import NaiveBayes
    >>> from skmultiflow.evaluation import EvaluateHoldout
    >>>
    >>> # Set the stream
    >>> stream = SEAGenerator(random_state=1)
    >>>
    >>> # Set the models
    >>> ht = HoeffdingTreeClassifier()
    >>> nb = NaiveBayes()
    >>>
    >>> evaluator = EvaluatePrequential(max_samples=24,
    >>>                                 max_time=1000,
    >>>                                 show_plot=True,
    >>>                                 metrics=['accuracy', 'kappa'])
    >>>
    >>> # Run evaluation
    >>> evaluator.evaluate(stream=stream, model=[ht, nb], model_names=['HT', 'NB'])

    >>> # The third example demonstrates how to evaluate one model
    >>> # and visualize the predictions using data points.
    >>> # Note: You can not in this case compare multiple models
    >>> from skmultiflow.data import SEAGenerator
    >>> from skmultiflow.trees import HoeffdingTreeClassifier
    >>> from skmultiflow.evaluation import EvaluatePrequential
    >>> # Set the stream
    >>> stream = SEAGenerator(random_state=1)
    >>> # Set the model
    >>> ht = HoeffdingTreeClassifier()
    >>> # Set the evaluator
    >>> evaluator = EvaluatePrequential(max_samples=200,
    >>>                                 n_wait=1,
    >>>                                 pretrain_size=1,
    >>>                                 max_time=1000,
    >>>                                 show_plot=True,
    >>>                                 metrics=['accuracy'],
    >>>                                 data_points_for_classification=True)
    >>>
    >>> # Run evaluation
    >>> evaluator.evaluate(stream=stream, model=ht, model_names=['HT'])

    """

    def __init__(self,
                 max_timestamps=24,
                 batch_size=1,
                 pretrain_timestamps=3,
                 max_time=float("inf"),
                 metrics=None,
                 output_file=None,
                 show_plot=False,
                 restart_stream=True):

        super().__init__()
        self._method = 'prequential'
        self.max_timestamps = max_timestamps
        self.pretrain_timestamps = pretrain_timestamps
        self.batch_size = batch_size
        self.max_time = max_time
        self.output_file = output_file
        self.show_plot = show_plot

        if metrics is None:
            self.metrics = [constants.DATA_POINTS]
        else:
            if isinstance(metrics, list):
                self.metrics = metrics
            else:
                raise ValueError("Attribute 'metrics' must be 'None' or 'list', passed {}".format(type(metrics)))

        self.restart_stream = restart_stream

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

        print("Will check configuration: {}".format(self._check_configuration()))
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
        print('Prequential Evaluation')
        print('Evaluating {} target(s).'.format(self.stream.n_targets))

        actual_max_timestamps = self.stream.n_remaining_timestamps()
        if actual_max_timestamps == -1 or actual_max_timestamps > self.max_timestamps:
            actual_max_timestamps = self.max_timestamps

        last_timestamp = None
        first_run = True

        print("evaluate_tprequential #1 {} > 0  -> {}".format(self.pretrain_timestamps, self.pretrain_timestamps > 0))
        if self.pretrain_timestamps > 0:
            while(self.global_timestamp_count < self.pretrain_timestamps):
                #TODO we shall cache by timestamps and evaluate all at once when timestamp changes


                if(first_run):
                    ta, Xa, ya = self.stream.next_sample(2)
                    for idx in range(0, 2):
                        t, X, y = ta[idx], Xa[idx], ya[idx]
                        if(idx ==0):
                            last_timestamp = t
                            first_run = False
                        else:
                            if(last_timestamp != t):
                                self.global_timestamp_count = self.global_timestamp_count + 1
                                last_timestamp = t

                    for i in range(self.n_models):
                        self.running_time_measurements[i].compute_training_time_begin()
                        self.model[i].partial_fit(t=t, X=X, y=y)
                        self.running_time_measurements[i].compute_training_time_end()
                        self.running_time_measurements[i].update_time_measurements(self.pretrain_timestamps)



                else:
                    t, X, y = self.stream.next_sample(1)
                    for i in range(self.n_models):
                        self.running_time_measurements[i].compute_training_time_begin()
                        self.model[i].partial_fit(t=t, X=X, y=y)
                        self.running_time_measurements[i].compute_training_time_end()
                        self.running_time_measurements[i].update_time_measurements(self.pretrain_timestamps)
                    if(last_timestamp != t):
                        self.global_timestamp_count = self.global_timestamp_count + 1
                        last_timestamp = t

        buffer_t = []
        buffer_X = []
        buffer_y = []
        update_count = 0
        print('Evaluating...')

        print("evaluate_tprequential #2 {}<{}  -> {}".format(self.global_timestamp_count, actual_max_timestamps, (self.global_timestamp_count < actual_max_timestamps)))
        print("evaluate_tprequential #3 self.stream.has_more_samples()  -> {}".format(self.stream.has_more_samples()))

        while ((self.global_timestamp_count < actual_max_timestamps) & (self._end_time - self._start_time < self.max_time)
               & (self.stream.has_more_samples())):
            try:
                t, X, y = self.stream.next_sample(1)
                if(last_timestamp != t):
                    self.global_timestamp_count = self.global_timestamp_count + 1
                    last_timestamp = t
                    self._update_metrics()
                    for idx in range(0, len(buffer_t)):
                        bt = buffer_t[idx]
                        bX = buffer_X[idx]
                        by = buffer_y[idx]
                        for i in range(self.n_models):
                            self.running_time_measurements[i].compute_training_time_begin()
                            self.model[i].partial_fit(bt, bX, by)
                            self.running_time_measurements[i].compute_training_time_end()
                            # Update total running time
                            self.running_time_measurements[i].update_time_measurements(self.batch_size)
                    buffer_t = []
                    buffer_X = []
                    buffer_y = []


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
                            raise TypeError("Unexpected prediction value from {}".format(type(self.model[i]).__name__))

                    for j in range(self.n_models):
                        for i in range(len(prediction[0])):
                            self.mean_eval_measurements[j].add_result(y[i], prediction[j][i])
                            self.current_eval_measurements[j].add_result(y[i], prediction[j][i])
                    self._check_progress(actual_max_timestamps)

                    # Train
                    ## add to buffer until a timestamp change is detected, to avoid asymetries due to time ordering
                    buffer_t.append(t)
                    buffer_X.append(X)
                    buffer_y.append(y)

                self._end_time = timer()
            except BaseException as exc:
                print(exc)
                if exc is KeyboardInterrupt:
                    self._update_metrics()
                break

        # Flush file buffer, in case it contains data
        self._flush_file_buffer()

        if len(set(self.metrics).difference({constants.DATA_POINTS})) > 0:
            self.evaluation_summary()
        else:
            print('Done')

        if self.restart_stream:
            self.stream.restart()

        return self.model

    def partial_fit(self, t, X, y, sample_weight=None):
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
                self.model[i].partial_fit(t=t, X=X, y=y, sample_weight=sample_weight)
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
