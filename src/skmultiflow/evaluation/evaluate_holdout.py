import os
import warnings
import logging
from timeit import default_timer as timer
from skmultiflow.evaluation.base_evaluator import StreamEvaluator


class EvaluateHoldout(StreamEvaluator):
    """ EvaluateHoldout
    
    The holdout evaluation method, or periodic holdout evaluation method, analyses 
    each arriving sample, without computing performance metrics, nor predicting 
    labels or regressing values, but by updating its statistics.
    
    The performance evaluation happens at every n_wait analysed samples, at which 
    moment the evaluator will test the learners performance on a test set, formed 
    by yet unseen samples, which will be used to evaluate performance, but not to 
    train the model. 
    
    It's possible to use the same test set for every test made, but it's also 
    possible to dynamically create test sets, so that they differ from each other. 
    If dynamic test sets are enabled, we use the data stream to create test sets 
    on the go. This process is more likely to generate test sets that follow the 
    current concept, in comparison to static test sets.
    
    Thus, if concept drift is known to be present in the dataset/generator enabling 
    dynamic test sets is highly recommended. If no concept drift is expected, 
    disabling this parameter will speed up the evaluation process.
    
    Parameters
    ----------
    n_wait: int (Default: 10000)
        The number of samples to process between each test. Also defines when to update the plot if `show_plot=True`.

    max_samples: int (Default: 100000)
        The maximum number of samples to process during the evaluation.
    
    batch_size: int (Default: 1)
        The number of samples to pass at a time to the model(s).

    max_time: float (Default: float("inf"))
        The maximum duration of the simulation (in seconds).
    
    metrics: list, optional (Default: ['performance'])
        The list of metrics to track during the evaluation. Also defines the metrics that will be displayed in plots
        and/or logged into the output file. Valid options are 'performance', 'kappa', 'kappa_t', 'kappa_m',
        'hamming_score', 'hamming_loss', 'exact_match', 'j_index', 'mean_square_error', 'mean_absolute_error',
        'true_vs_predicts'.

    output_file: string, optional (Default: None)
        File name to save the summary of the evaluation.

    show_plot: bool (Default: False)
        If True, a plot will show the progress of the evaluation. Warning: Plotting will slow down the evaluation
        process.

    restart_stream: bool, optional (default=True)
        If True, the stream is restarted once the evaluation is complete.

    test_size: int (Default: 20000)
        The size of the test set.

    dynamic_test_set: bool (Default: False)
        If True, will continuously change the test set, otherwise will use the same test set for all tests.
    
    Notes
    -----
    It's important to note that testing the model too often, which means choosing 
    a `n_wait` parameter too small, will significantly slow the evaluation process,
    depending on the test size. 
    
    This evaluator accepts to types of evaluation processes. It can either evaluate 
    a single learner while computing its metrics or it can evaluate multiple learners 
    at a time, as a means of comparing different approaches to the same problem.
    
    Examples
    --------
    >>> # The first example demonstrates how to use the evaluator to evaluate one learner
    >>> from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
    >>> from skmultiflow.core.pipeline import Pipeline
    >>> from skmultiflow.data.file_stream import FileStream
    >>> from skmultiflow.evaluation.evaluate_holdout import EvaluateHoldout
    >>> # Setup the File Stream
    >>> stream = FileStream("skmultiflow/data/datasets/covtype.csv", -1, 1)
    >>> stream.prepare_for_use()
    >>> # Setup the classifier
    >>> classifier = PassiveAggressiveClassifier()
    >>> # Setup the pipeline
    >>> pipe = Pipeline([('Classifier', classifier)])
    >>> # Setup the evaluator
    >>> evaluator = EvaluateHoldout(max_samples=100000, batch_size=1, n_wait=10000, max_time=1000,
    >>>                             output_file=None, show_plot=True, metrics=['kappa', 'performance'],
    >>>                             test_size=5000, dynamic_test_set=True)
    >>> # Evaluate
    >>> evaluator.evaluate(stream=stream, model=pipe)
    
    >>> # The second example will demonstrate how to compare two classifiers with
    >>> # the EvaluateHoldout
    >>> from skmultiflow.data import WaveformGenerator
    >>> from sklearn.linear_model.stochastic_gradient import SGDClassifier
    >>> from skmultiflow.evaluation import EvaluateHoldout
    >>> from skmultiflow.lazy import KNNAdwin
    >>> stream = WaveformGenerator()
    >>> stream.prepare_for_use()
    >>> clf_one = SGDClassifier()
    >>> clf_two = KNNAdwin(n_neighbors=8,max_window_size=2000)
    >>> classifier = [clf_one, clf_two]
    >>> evaluator = EvaluateHoldout(test_size=5000, dynamic_test_set=True, max_samples=100000, batch_size=1,
    >>>                             n_wait=10000, max_time=1000, output_file=None, show_plot=True,
    >>>                             metrics=['kappa', 'performance'])
    >>> evaluator.evaluate(stream=stream, model=classifier)
    
    """

    def __init__(self,
                 n_wait=10000,
                 max_samples=100000,
                 batch_size=1,
                 max_time=float("inf"),
                 metrics=None,
                 output_file=None,
                 show_plot=False,
                 restart_stream=True,
                 test_size=5000,
                 dynamic_test_set=False):

        super().__init__()
        self._method = 'holdout'
        self.n_wait = n_wait
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.max_time = max_time
        self.output_file = output_file
        self.show_plot = show_plot
        if metrics is None:
            self.metrics = [self.PERFORMANCE, self.KAPPA]
        else:
            self.metrics = metrics
        self.restart_stream = restart_stream
        # Holdout parameters
        self.dynamic_test_set = dynamic_test_set
        if test_size < 0:
            raise ValueError('test_size has to be greater than 0.')
        else:
            self.test_size = test_size
        self.n_sliding = test_size

        warnings.filterwarnings("ignore", ".*invalid value encountered in true_divide.*")
        warnings.filterwarnings("ignore", ".*Passing 1d.*")

    def evaluate(self, stream, model, model_names=None):
        """ evaluate
        
        Parameters
        ---------
        stream: A stream (an extension from BaseInstanceStream) 
            The stream from which to draw the samples. 
        
        model: A learner (an extension from BaseClassifier) or a list of learners.
            The learner or learners on which to train the model and measure the 
            performance metrics.

        model_names: list, optional (Default=None)
            A list with the names of the learners.
            
        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifier's at the end of the evaluation process.
        
        """
        # First off we need to verify if this is a simple evaluation task or a comparison between learners task.
        self._init_evaluation(model=model, stream=stream, model_names=model_names)

        if self._check_configuration():
            self._reset_globals()
            # Initialize metrics and outputs (plots, log files, ...)
            self._init_metrics()
            self._init_plot()
            self._init_file()

            self.model = self._periodic_holdout()

            if self.show_plot:
                self.visualizer.hold()

            return self.model

    def _periodic_holdout(self):
        """ Method to control the holdout evaluation.
             
        Returns
        -------
        BaseClassifier extension or list of BaseClassifier extensions
            The trained classifiers. 
        
        Notes
        -----
        The classifier parameter should be an extension from the BaseClassifier. In 
        the future, when BaseRegressor is created, it could be an axtension from that 
        class as well.
        
        """
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        init_time = timer()
        end_time = timer()
        logging.info('Holdout Evaluation')
        logging.info('Evaluating %s target(s).', str(self.stream.n_targets))

        n_samples = self.stream.n_remaining_samples()
        if n_samples == -1 or n_samples > self.max_samples:
            n_samples = self.max_samples

        first_run = True
        # if self.pretrain_size > 0:
        #     logging.info('Pre-training on %s samples.', str(self.pretrain_size))
        #     X, y = self.stream.next_sample(self.pretrain_size)
        #     for i in range(self.n_models):
        #         if self._task_type != EvaluateHoldout._REGRESSION:
        #             self.model[i].partial_fit(X=X, y=y, classes=self.stream.target_values)
        #         else:
        #             self.model[i].partial_fit(X=X, y=y)
        #     self.global_sample_count += self.pretrain_size
        #     first_run = False
        # else:
        #     logging.info('Pre-training on 1 sample.')   # TODO Confirm if needed
        #     X, y = self.stream.next_sample()
        #     for i in range(self.n_models):
        #         if self.task_type != 'regression':
        #             self.model[i].partial_fit(X, y, self.stream.get_targets())
        #         else:
        #             self.model[i].partial_fit(X, y)
        #     first_run = False

        if not self.dynamic_test_set:
            logging.info('Separating %s holdout samples.', str(self.test_size))
            self.X_test, self.y_test = self.stream.next_sample(self.test_size)
            self.global_sample_count += self.test_size

        performance_sampling_cnt = 0
        logging.info('Evaluating...')
        while ((self.global_sample_count < self.max_samples) & (end_time - init_time < self.max_time)
               & (self.stream.has_more_samples())):
            try:
                X, y = self.stream.next_sample(self.batch_size)

                if X is not None and y is not None:
                    self.global_sample_count += self.batch_size

                    # Train
                    if first_run:
                        for i in range(self.n_models):
                            if self._task_type != EvaluateHoldout.REGRESSION:
                                self.model[i].partial_fit(X, y, self.stream.target_values)
                            else:
                                self.model[i].partial_fit(X, y)
                        first_run = False
                    else:
                        for i in range(self.n_models):
                            self.model[i].partial_fit(X, y)

                    self._check_progress(n_samples)   # TODO Confirm place

                    # Test on holdout set
                    if self.dynamic_test_set:
                        perform_test = self.global_sample_count == (self.n_wait * (performance_sampling_cnt + 1)
                                                                    + (self.test_size * performance_sampling_cnt))
                    else:
                        perform_test = (self.global_sample_count - self.test_size) % self.n_wait == 0

                    if perform_test | (self.global_sample_count >= self.max_samples):

                        if self.dynamic_test_set:
                            logging.info('Separating %s holdout samples.', str(self.test_size))
                            self.X_test, self.y_test = self.stream.next_sample(self.test_size)
                            self.global_sample_count += self.test_size

                        # Test
                        if (self.X_test is not None) and (self.y_test is not None):
                            prediction = [[] for _ in range(self.n_models)]
                            for i in range(self.n_models):
                                try:
                                    prediction[i].extend(self.model[i].predict(self.X_test))
                                except TypeError:
                                    raise TypeError("Unexpected prediction value from {}"
                                                    .format(type(self.model[i]).__name__))
                            if prediction is not None:
                                for j in range(self.n_models):
                                    for i in range(len(prediction[0])):
                                        if self._task_type == EvaluateHoldout.CLASSIFICATION:
                                            self.global_classification_metrics[j].add_result(self.y_test[i],
                                                                                             prediction[j][i])
                                            self.partial_classification_metrics[j].add_result(self.y_test[i],
                                                                                              prediction[j][i])
                                        else:
                                            self.global_classification_metrics[j].add_result(self.y_test[i],
                                                                                             prediction[j][i])
                                            self.partial_classification_metrics[j].add_result(self.y_test[i],
                                                                                              prediction[j][i])
                                self._update_metrics()
                            performance_sampling_cnt += 1

                end_time = timer()
            except BaseException as exc:
                print(exc)
                if exc is KeyboardInterrupt:
                    self._update_metrics()
                break

        if end_time - init_time > self.max_time:
            logging.info('Time limit reached. Evaluation stopped.')
            logging.info('Evaluation time: {} s'.format(self.max_time))
        else:
            logging.info('Evaluation time: {:.3f} s'.format(end_time - init_time))
        logging.info('Total samples: {}'.format(self.global_sample_count))
        logging.info('Global performance:')
        for i in range(self.n_models):
            if 'performance' in self.metrics:
                logging.info('{} - Accuracy     : {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_performance()))
            if 'kappa' in self.metrics:
                logging.info('{} - Kappa        : {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_kappa()))
            if 'kappa_t' in self.metrics:
                logging.info('{} - Kappa T      : {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_kappa_t()))
            if 'kappa_m' in self.metrics:
                logging.info('{} - Kappa M      : {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_kappa_m()))
            if 'hamming_score' in self.metrics:
                logging.info('{} - Hamming score: {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_hamming_score()))
            if 'hamming_loss' in self.metrics:
                logging.info('{} - Hamming loss : {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_hamming_loss()))
            if 'exact_match' in self.metrics:
                logging.info('{} - Exact matches: {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_exact_match()))
            if 'j_index' in self.metrics:
                logging.info('{} - j index      : {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_j_index()))
            if 'mean_square_error' in self.metrics:
                logging.info('{} - MSE          : {:.3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_mean_square_error()))
            if 'mean_absolute_error' in self.metrics:
                logging.info('{} - MAE          : {:3f}'.format(
                    self.model_names[i], self.global_classification_metrics[i].get_average_error()))

        if self.restart_stream:
            self.stream.restart()

        return self.model

    def partial_fit(self, X, y, classes=None, weight=None):
        """ partial_fit
        
        Partially fit all the learners on the given data.
        
        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            The data upon which the algorithm will create its model.
            
        y: Array-like
            An array-like containing the classification targets for all samples in X.
            
        classes: list
            Stores all the classes that may be encountered during the classification task.

        weight: Array-like
            Instance weight. If not provided, uniform weights are assumed.
        
        Returns
        -------
        EvaluateHoldout
            self
         
        """
        if self.model is not None:
            for i in range(self.n_models):
                self.model[i].partial_fit(X, y, classes, weight)
            return self
        else:
            return self

    def predict(self, X):
        """ predict

        Predicts the labels of the X samples, by calling the predict 
        function of all the learners.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.

        Returns
        -------
        list
            A list containing the predicted labels for all instances in X in 
            all learners.

        """
        predictions = None
        if self.model is not None:
            predictions = []
            for i in range(self.n_models):
                predictions.append(self.model[i].predict(X))

        return predictions

    def _check_progress(self, n_samples):
        progress = self.global_sample_count - self.batch_size

        # Update progress
        if (progress % (n_samples // 20)) == 0:
            logging.info('{}%'.format(progress // (n_samples / 20) * 5))

    def set_params(self, parameter_dict):
        """ set_params
        
        This function allows the users to change some of the evaluator's parameters, 
        by passing a dictionary where keys are the parameters names, and values are 
        the new parameters' values.
        
        Parameters
        ----------
        dict: Dictionary
            A dictionary where the keys are the names of attributes the user 
            wants to change, and the values are the new values of those attributes.
             
        """
        for name, value in parameter_dict.items():
            if name == 'n_wait':
                self.n_wait = value
            elif name == 'max_samples':
                self.max_samples = value
            elif name == 'max_time':
                self.max_time = value
            elif name == 'output_file':
                self.output_file = value
            elif name == 'batch_size':
                self.batch_size = value
            elif name == 'pretrain_size':
                self.pretrain_size = value
            elif name == 'test_size':
                self.test_size = value

    def get_info(self):
        filename = "None"
        if self.output_file is not None:
            path, filename = os.path.split(self.output_file)
        return 'Holdout Evaluator: n_wait: ' + str(self.n_wait) + \
               ' - max_samples: ' + str(self.max_samples) + \
               ' - max_time: ' + str(self.max_time) + \
               ' - output_file: ' + filename + \
               ' - batch_size: ' + str(self.batch_size) + \
               ' - task_type: ' + self._task_type + \
               ' - show_plot: ' + ('True' if self.show_plot else 'False') + \
               ' - metrics: ' + (str(self.metrics) if self.metrics is not None else 'None') + \
               ' - test_size: ' + str(self.test_size) + \
               ' - dynamic_test_set: ' + ('True' if self.dynamic_test_set else 'False')
