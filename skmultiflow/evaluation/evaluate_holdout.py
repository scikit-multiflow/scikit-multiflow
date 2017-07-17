__author__ = 'Guilherme Matsumoto'

import warnings
import logging
from timeit import default_timer as timer
from skmultiflow.evaluation.base_evaluator import BaseEvaluator
from skmultiflow.evaluation.measure_collection import ClassificationMeasurements, WindowClassificationMeasurements, RegressionMeasurements, WindowRegressionMeasurements, MultiOutputMeasurements, WindowMultiOutputMeasurements
from skmultiflow.core.utils.utils import dict_to_tuple_list
from skmultiflow.visualization.evaluation_visualizer import EvaluationVisualizer


class EvaluateHoldout(BaseEvaluator):
    def __init__(self, n_wait=10000, max_instances=100000, max_time=float("inf"), output_file=None,
                 batch_size=1, pretrain_size=200, test_size=20000, task_type='classification', show_plot=False,
                 plot_options=None, dynamic_test_set=False):
        '''
        
        :param n_wait: 
        :param max_instances: 
        :param max_time: 
        :param output_file: 
        :param batch_size: 
        :param pretrain_size: 
        :param task_type: 
        :param show_plot: 
        :param plot_options: 
        '''
        PLOT_TYPES = ['performance', 'kappa', 'scatter', 'hamming_score', 'hamming_loss', 'exact_match', 'j_index',
                      'mean_square_error', 'mean_absolute_error', 'true_vs_predicts', 'kappa_t', 'kappa_m']
        TASK_TYPES = ['classification', 'regression', 'multi_output']

        super().__init__()
        self.n_wait = n_wait
        self.max_instances = max_instances
        self.max_time = max_time
        self.batch_size = batch_size
        self.pretrain_size = pretrain_size
        self.test_size = test_size
        self.classifier = None
        self.stream = None
        self.output_file = output_file
        self.visualizer = None
        self.X_test = None
        self.y_test = None
        self.dynamic_test_set = dynamic_test_set

        if self.test_size < 0:
            raise ValueError('test_size has to be greater than 0.')

        # plotting configs
        self.task_type = task_type.lower()
        if self.task_type not in TASK_TYPES:
            raise ValueError('Task type not supported.')
        self.show_plot = show_plot
        self.plot_options = None
        if self.show_plot is True and plot_options is None:
            if self.task_type == 'classification':
                self.plot_options = ['performance', 'kappa']
            elif self.task_type == 'regression':
                self.plot_options = ['mean_square_error', 'true_vs_predict']
            elif self.task_type == 'multi_output':
                self.plot_options = ['hamming_score', 'exact_match', 'j_index']
        elif self.show_plot is True and plot_options is not None:
            self.plot_options = [x.lower() for x in plot_options]
        for i in range(len(self.plot_options)):
            if self.plot_options[i] not in PLOT_TYPES:
                raise ValueError(str(self.plot_options[i]) + ': Plot type not supported.')

        # metrics
        self.global_classification_metrics = None
        self.partial_classification_metrics = None
        if self.task_type in ['classification']:
            self.global_classification_metrics = ClassificationMeasurements()
            self.partial_classification_metrics = WindowClassificationMeasurements(window_size=self.test_size)
        elif self.task_type in ['multi_output']:
            self.global_classification_metrics = MultiOutputMeasurements()
            self.partial_classification_metrics = WindowMultiOutputMeasurements(window_size=self.test_size)
        elif self.task_type in ['regression']:
            self.global_classification_metrics = RegressionMeasurements()
            self.partial_classification_metrics = WindowRegressionMeasurements(window_size=self.test_size)

        self.global_sample_count = 0

        warnings.filterwarnings("ignore", ".*invalid value encountered in true_divide.*")

    def eval(self, stream, classifier):
        if self.show_plot:
            self.start_plot(self.n_wait, stream.get_plot_name())
        self._reset_globals()
        self.classifier = classifier
        self.stream = stream
        self.classifier = self.periodic_holdout(stream, classifier)
        if self.show_plot:
            self.visualizer.hold()
        return self.classifier

    def periodic_holdout(self, stream=None, classifier=None):
        logging.basicConfig(format='%(message)s', level=logging.INFO)
        init_time = timer()
        end_time = timer()
        if classifier is not None:
            self.classifier = classifier
        if stream is not None:
            self.stream = stream
        self._reset_globals()
        prediction = None
        logging.info('Holdout Evaluation')
        logging.info('Generating %s targets.', str(self.stream.get_num_targets()))

        rest = self.stream.estimated_remaining_instances() if (self.stream.estimated_remaining_instances() != -1 and
                                                               self.stream.estimated_remaining_instances() <=
                                                               self.max_instances) \
            else self.max_instances

        if self.output_file is not None:
            with open(self.output_file, 'w+') as f:
                f.write("# SETUP BEGIN")
                if hasattr(self.stream, 'get_info'):
                    f.write("\n# " + self.stream.get_info())
                if hasattr(self.classifier, 'get_info'):
                    f.write("\n# " + self.classifier.get_info())
                f.write("\n# " + self.get_info())
                f.write("\n# SETUP END")
                header = '\nx_count'
                if 'performance' in self.plot_options:
                    header += ',global_performance,sliding_window_performance'
                if 'kappa' in self.plot_options:
                    header += ',global_kappa,sliding_window_kappa'
                if 'kappa_t' in self.plot_options:
                    header += ',global_kappa_t,sliding_window_kappa_t'
                if 'kappa_m' in self.plot_options:
                    header += ',global_kappa_m,sliding_window_kappa_m'
                if 'scatter' in self.plot_options:
                    header += ',true_label,prediction'
                if 'hamming_score' in self.plot_options:
                    header += ',global_hamming_score,sliding_window_hamming_score'
                if 'hamming_loss' in self.plot_options:
                    header += ',global_hamming_loss,sliding_window_hamming_loss'
                if 'exact_match' in self.plot_options:
                    header += ',global_exact_match,sliding_window_exact_match'
                if 'j_index' in self.plot_options:
                    header += ',global_j_index,sliding_window_j_index'
                if 'mean_square_error' in self.plot_options:
                    header += ',global_mse,sliding_window_mse'
                if 'mean_absolute_error' in self.plot_options:
                    header += ',global_mae,sliding_window_mae'
                    # if 'true_vs_predicts' in self.plot_options:
                    # header += ',true_label,prediction'
                f.write(header)

        first_run = True
        if (self.pretrain_size > 0):
            logging.info('Pretraining on %s samples.', str(self.pretrain_size))
            X, y = self.stream.next_instance(self.pretrain_size)
            self.classifier.partial_fit(X, y, self.stream.get_classes())
            first_run = False
        else:
            logging.info('Pretraining on 1 sample.')
            X, y = self.stream.next_instance()
            self.classifier.partial_fit(X, y, self.stream.get_classes())
            first_run = False

        if not self.dynamic_test_set:
            logging.info('Separating %s static holdout samples.', str(self.test_size))
            self.X_test, self.y_test = self.stream.next_instance(self.test_size)

        before_count = 0
        logging.info('Evaluating...')
        while ((self.global_sample_count < self.max_instances) & (end_time - init_time < self.max_time)
                   & (self.stream.has_more_instances())):
            try:
                X, y = self.stream.next_instance(self.batch_size)
                if X is not None and y is not None:
                    self.global_sample_count += self.batch_size
                    if first_run:
                        self.classifier.partial_fit(X, y, self.stream.get_classes())
                        first_run = False
                    else:
                        self.classifier.partial_fit(X, y)

                    nul_count = self.global_sample_count - self.batch_size
                    for i in range(self.batch_size):
                        if ((nul_count + i + 1) % (rest / 20)) == 0:
                            logging.info('%s%%', str(((nul_count + i + 1) // (rest / 20)) * 5))

                    # Test on holdout set
                    if ((self.global_sample_count % self.n_wait) == 0 | (
                                self.global_sample_count >= self.max_instances) |
                        (self.global_sample_count / self.n_wait > before_count + 1)):

                        if self.dynamic_test_set:
                            logging.info('Separating %s dynamic holdout samples.', str(self.test_size))
                            self.X_test, self.y_test = self.stream.next_instance(self.test_size)

                        if (self.X_test is not None) and (self.y_test is not None):
                            logging.info('Testing model on %s samples.', str(self.test_size))

                            for i in range(self.test_size):
                                prediction = self.classifier.predict([self.X_test[i]])

                            #for i in range(len(prediction)):
                                self.global_classification_metrics.add_result(self.y_test[i], prediction)
                                self.partial_classification_metrics.add_result(self.y_test[i], prediction)
                            before_count += 1
                            self.update_metrics()

                end_time = timer()
            except BaseException as exc:
                if exc is KeyboardInterrupt:
                    if self.show_scatter_points:
                        self.update_metrics()
                    else:
                        self.update_metrics()
                break

        if (end_time - init_time > self.max_time):
            logging.info('\nTime limit reached. Evaluation stopped.')
            logging.info('Evaluation time: %s s', str(self.max_time))
        else:
            logging.info('\nEvaluation time: %s s', str(round(end_time - init_time, 3)))
        logging.info('Total instances: %s', str(self.global_sample_count))

        if 'performance' in self.plot_options:
            logging.info('Global accuracy: %s', str(round(self.global_classification_metrics.get_performance(), 3)))
        if 'kappa' in self.plot_options:
            logging.info('Global kappa: %s', str(round(self.global_classification_metrics.get_kappa(), 3)))
        if 'kappa_t' in self.plot_options:
            logging.info('Global kappa T: %s', str(round(self.global_classification_metrics.get_kappa_t(), 3)))
        if 'kappa_m' in self.plot_options:
            logging.info('Global kappa M: %s', str(round(self.global_classification_metrics.get_kappa_m(), 3)))
        if 'scatter' in self.plot_options:
            pass
        if 'hamming_score' in self.plot_options:
            logging.info('Global hamming score: %s',
                         str(round(self.global_classification_metrics.get_hamming_score(), 3)))
        if 'hamming_loss' in self.plot_options:
            logging.info('Global hamming loss: %s',
                         str(round(self.global_classification_metrics.get_hamming_loss(), 3)))
        if 'exact_match' in self.plot_options:
            logging.info('Global exact matches: %s',
                         str(round(self.global_classification_metrics.get_exact_match(), 3)))
        if 'j_index' in self.plot_options:
            logging.info('Global j index: %s', str(round(self.global_classification_metrics.get_j_index(), 3)))
        if 'mean_square_error' in self.plot_options:
            logging.info('Global MSE: %s', str(round(self.global_classification_metrics.get_mean_square_error(), 6)))
        if 'mean_absolute_error' in self.plot_options:
            logging.info('Global MAE: %s', str(round(self.global_classification_metrics.get_average_error(), 6)))
        if 'true_vs_predicts' in self.plot_options:
            pass

        return self.classifier

    def partial_fit(self, X, y):
        if self.classifier is not None:
            self.classifier.partial_fit(X, y)
            return self
        else:
            return self

    def predict(self, X):
        if self.classifier is not None:
            self.classifier.predict(X)
            return self
        else:
            return self

    def update_plot(self, current_x, new_points_dict):
        if self.output_file is not None:
            line = str(current_x)
            if 'classification' in self.plot_options:
                line += ',' + str(round(self.global_classification_metrics.get_performance(), 3))
                line += ',' + str(round(self.partial_classification_metrics.get_performance(), 3))
            if 'kappa' in self.plot_options:
                line += ',' + str(round(self.global_classification_metrics.get_kappa(), 3))
                line += ',' + str(round(self.partial_classification_metrics.get_kappa(), 3))
            if 'kappa_t' in self.plot_options:
                line += ',' + str(round(self.global_classification_metrics.get_kappa_t(), 3))
                line += ',' + str(round(self.partial_classification_metrics.get_kappa_t(), 3))
            if 'kappa_m' in self.plot_options:
                line += ',' + str(round(self.global_classification_metrics.get_kappa_m(), 3))
                line += ',' + str(round(self.partial_classification_metrics.get_kappa_m(), 3))
            if 'scatter' in self.plot_options:
                line += ',' + str(new_points_dict['scatter'][0]) + ',' + str(new_points_dict['scatter'][1])
            if 'hamming_score' in self.plot_options:
                line += ',' + str(round(self.global_classification_metrics.get_hamming_score() ,3))
                line += ',' + str(round(self.partial_classification_metrics.get_hamming_score(), 3))
            if 'hamming_loss' in self.plot_options:
                line += ',' + str(round(self.global_classification_metrics.get_hamming_loss() ,3))
                line += ',' + str(round(self.partial_classification_metrics.get_hamming_loss(), 3))
            if 'exact_match' in self.plot_options:
                line += ',' + str(round(self.global_classification_metrics.get_exact_match() ,3))
                line += ',' + str(round(self.partial_classification_metrics.get_exact_match(), 3))
            if 'j_index' in self.plot_options:
                line += ',' + str(round(self.global_classification_metrics.get_j_index() ,3))
                line += ',' + str(round(self.partial_classification_metrics.get_j_index(), 3))
            if 'mean_square_error' in self.plot_options:
                line += ',' + str(round(self.global_classification_metrics.get_mean_square_error(), 6))
                line += ',' + str(round(self.partial_classification_metrics.get_mean_square_error(), 6))
            if 'mean_absolute_error' in self.plot_options:
                line += ',' + str(round(self.global_classification_metrics.get_average_error(), 6))
                line += ',' + str(round(self.partial_classification_metrics.get_average_error(), 6))
            with open(self.output_file, 'a') as f:
                f.write('\n' + line)

        self.visualizer.on_new_train_step(current_x, new_points_dict)

    def update_metrics(self):
        """ Updates the metrics of interest.

            It's possible that cohen_kappa_score will return a NaN value, which happens if the predictions
            and the true labels are in perfect accordance, causing pe=1, which results in a division by 0.
            If this is detected the plot will assume it to be 1.

        :return: No return.
        """

        new_points_dict = {}

        if 'performance' in self.plot_options:
            new_points_dict['performance'] = [self.global_classification_metrics.get_performance(), self.partial_classification_metrics.get_performance()]
        if 'kappa' in self.plot_options:
            new_points_dict['kappa'] = [self.global_classification_metrics.get_kappa(), self.partial_classification_metrics.get_kappa()]
        if 'kappa_t' in self.plot_options:
            new_points_dict['kappa_t'] = [self.global_classification_metrics.get_kappa_t(), self.partial_classification_metrics.get_kappa_t()]
        if 'kappa_m' in self.plot_options:
            new_points_dict['kappa_m'] = [self.global_classification_metrics.get_kappa_m(), self.partial_classification_metrics.get_kappa_m()]
        if 'scatter' in self.plot_options:
            true, pred = self.global_classification_metrics.get_last()
            new_points_dict['scatter'] = [true, pred]
        if 'hamming_score' in self.plot_options:
            new_points_dict['hamming_score'] = [self.global_classification_metrics.get_hamming_score(), self.partial_classification_metrics.get_hamming_score()]
        if 'hamming_loss' in self.plot_options:
            new_points_dict['hamming_loss'] = [self.global_classification_metrics.get_hamming_loss(), self.partial_classification_metrics.get_hamming_loss()]
        if 'exact_match' in self.plot_options:
            new_points_dict['exact_match'] = [self.global_classification_metrics.get_exact_match(), self.partial_classification_metrics.get_exact_match()]
        if 'j_index' in self.plot_options:
            new_points_dict['j_index'] = [self.global_classification_metrics.get_j_index(), self.partial_classification_metrics.get_j_index()]
        if 'mean_square_error' in self.plot_options:
            new_points_dict['mean_square_error'] = [self.global_classification_metrics.get_mean_square_error(), self.partial_classification_metrics.get_mean_square_error()]
        if 'mean_absolute_error' in self.plot_options:
            new_points_dict['mean_absolute_error'] = [self.global_classification_metrics.get_average_error(), self.partial_classification_metrics.get_average_error()]
        if 'true_vs_predicts' in self.plot_options:
            true, pred = self.global_classification_metrics.get_last()
            new_points_dict['true_vs_predicts'] = [true, pred]
            #print(str(true) + ' ' + str(pred))
        if self.show_plot:
            self.update_plot(self.global_sample_count, new_points_dict)

    def set_params(self, dict):
        params_list = dict_to_tuple_list(dict)
        for name, value in params_list:
            if name == 'n_wait':
                self.n_wait = value
            elif name == 'max_instances':
                self.max_instances = value
            elif name == 'max_time':
                self.max_time = value
            elif name == 'output_file':
                self.output_file = value
            elif name == 'show_performance':
                self.show_performance = value
            elif name == 'batch_size':
                self.batch_size = value
            elif name == 'pretrain_size':
                self.pretrain_size = value
            elif name == 'show_kappa':
                self.show_kappa = value
            elif name == 'show_scatter_points':
                self.show_scatter_points = value

    def start_plot(self, n_wait, dataset_name):
        self.visualizer = EvaluationVisualizer(n_wait=n_wait, dataset_name=dataset_name, plots=self.plot_options)

    def _reset_globals(self):
        self.global_sample_count = 0

    def get_info(self):
        return 'Not implemented.'