import numpy as np
import time
from skmultiflow.metrics import ClassificationPerformanceEvaluator
from skmultiflow.metrics import WindowClassificationPerformanceEvaluator
from skmultiflow.metrics import MultiLabelClassificationPerformanceEvaluator
from skmultiflow.metrics import WindowMultiLabelClassificationPerformanceEvaluator
from skmultiflow.metrics import RegressionMeasurements
from skmultiflow.metrics import WindowRegressionMeasurements
from skmultiflow.metrics import MultiTargetRegressionMeasurements
from skmultiflow.metrics import WindowMultiTargetRegressionMeasurements
from skmultiflow.metrics import RunningTimeMeasurements


def test_classification_performance_evaluator():
    y_true = np.concatenate((np.ones(85), np.zeros(10), np.ones(5)))
    y_pred = np.concatenate((np.ones(90), np.zeros(10)))

    measurements = ClassificationPerformanceEvaluator()
    for i in range(len(y_true)):
        measurements.add_result(y_true[i], y_pred[i])

    expected_acc = 90/100
    assert expected_acc == measurements.accuracy_score()

    expected_kappa = (expected_acc - 0.82) / (1 - 0.82)
    assert np.isclose(expected_kappa, measurements.kappa_score())

    expected_kappa_m = (expected_acc - .9) / (1 - 0.9)
    assert np.isclose(expected_kappa_m, measurements.kappa_m_score())

    expected_kappa_t = (expected_acc - .97) / (1 - 0.97)
    assert expected_kappa_t == measurements.kappa_t_score()

    expected_precision = 85 / (85+5)
    assert np.isclose(expected_precision, measurements.precision_score())

    expected_recall = 85 / (85+5)
    assert np.isclose(expected_recall, measurements.recall_score())

    expected_f1_score = 2 * ((expected_precision * expected_recall) / (expected_precision + expected_recall))
    assert np.isclose(expected_f1_score, measurements.f1_score())

    expected_g_mean = np.sqrt((5 / (5 + 5)) * expected_recall)
    assert np.isclose(expected_g_mean, measurements.geometric_mean_score())

    expected_info = ("ClassificationPerformanceEvaluator(n_classes=2, "
                     "n_samples=100, accuracy_score=0.900000, "
                     "kappa_m_score=0.000000, kappa_t_score=-2.333333, "
                     "kappa_m_score=0.000000, precision_score=0.944444, "
                     "recall_score=0.944444, f1_score=0.944444, "
                     "geometric_mean_score=0.687184, majority_class=1)")
    assert expected_info == measurements.get_info()

    expected_last = (1.0, 0.0)
    assert expected_last == measurements.get_last()

    expected_majority_class = 1
    assert expected_majority_class == measurements.majority_class()

    measurements.reset()
    assert measurements.n_samples == 0


def test_window_classification_performance_evaluator():
    y_true = np.concatenate((np.ones(85), np.zeros(10), np.ones(5)))
    y_pred = np.concatenate((np.ones(90), np.zeros(10)))

    measurements = WindowClassificationPerformanceEvaluator(window_size=20)
    for i in range(len(y_true)):
        measurements.add_result(y_true[i], y_pred[i])

    expected_acc = 10/20
    assert expected_acc == measurements.accuracy_score()

    expected_kappa = 0.0
    assert np.isclose(expected_kappa, measurements.kappa_score())

    expected_kappa_m = 0.28571428571428575
    assert np.isclose(expected_kappa_m, measurements.kappa_m_score())

    expected_kappa_t = -4.0
    assert np.isclose(expected_kappa_t, measurements.kappa_t_score())

    expected_precision = 5 / (5 + 5)
    assert np.isclose(expected_precision, measurements.precision_score())

    expected_recall = 5 / (5 + 5)
    assert np.isclose(expected_recall, measurements.recall_score())

    expected_f1_score = 2 * ((expected_precision * expected_recall) / (expected_precision + expected_recall))
    assert np.isclose(expected_f1_score, measurements.f1_score())

    expected_g_mean = np.sqrt((5 / (5 + 5)) * expected_recall)
    assert np.isclose(expected_g_mean, measurements.geometric_mean_score())

    expected_info = ("WindowClassificationPerformanceEvaluator(n_classes=2, "
                     "window_size=20, n_samples=20, accuracy_score=0.500000, "
                     "kappa_m_score=0.285714, kappa_t_score=-4.000000, "
                     "kappa_m_score=0.285714, precision_score=0.500000, "
                     "recall_score=0.500000, f1_score=0.500000, "
                     "geometric_mean_score=0.500000, majority_class=0)")

    expected_last = (1.0, 0.0)
    assert expected_last == measurements.get_last()

    expected_majority_class = 0
    assert expected_majority_class == measurements.majority_class()

    measurements.reset()
    assert measurements.n_samples == 0


def test_multi_label_classification_performance_evaluator():
    y_0 = np.ones(100)
    y_1 = np.concatenate((np.ones(90), np.zeros(10)))
    y_2 = np.concatenate((np.ones(85), np.zeros(10), np.ones(5)))
    y_true = np.ones((100, 3))
    y_pred = np.vstack((y_0, y_1, y_2)).T

    measurements = MultiLabelClassificationPerformanceEvaluator()
    for i in range(len(y_true)):
        measurements.add_result(y_true[i], y_pred[i])

    expected_acc = 0.85
    assert np.isclose(expected_acc, measurements.exact_match_score())

    expected_hamming_score = 1 - 0.06666666666666667
    assert np.isclose(expected_hamming_score, measurements.hamming_score())

    expected_hamming_loss = 0.06666666666666667
    assert np.isclose(expected_hamming_loss, measurements.hamming_loss_score())

    expected_jaccard_index = 0.9333333333333332
    assert np.isclose(expected_jaccard_index, measurements.jaccard_score())

    expected_info = ("MultiLabelClassificationPerformanceEvaluator(n_labels=3,"
                     " n_samples=100, hamming_score=0.933333, "
                     "hamming_loss_score=0.066667, exact_match_score=0.850000,"
                     " jaccard_score=0.933333)")
    assert expected_info == measurements.get_info()

    expected_last_true = (1.0, 1.0, 1.0)
    expected_last_pred = (1.0, 0.0, 1.0)
    assert np.alltrue(expected_last_true == measurements.get_last()[0])
    assert np.alltrue(expected_last_pred == measurements.get_last()[1])

    measurements.reset()
    assert measurements.n_samples == 0


def test_window_multi_label_classification_performance_evaluator():
    y_0 = np.ones(100)
    y_1 = np.concatenate((np.ones(90), np.zeros(10)))
    y_2 = np.concatenate((np.ones(85), np.zeros(10), np.ones(5)))
    y_true = np.ones((100, 3))
    y_pred = np.vstack((y_0, y_1, y_2)).T

    measurements = WindowMultiLabelClassificationPerformanceEvaluator(window_size=20)
    for i in range(len(y_true)):
        measurements.add_result(y_true[i], y_pred[i])

    expected_acc = 0.25
    assert np.isclose(expected_acc, measurements.exact_match_score())

    expected_hamming_score = 1 - 0.33333333333333337
    assert np.isclose(expected_hamming_score, measurements.hamming_score())

    expected_hamming_loss = 0.33333333333333337
    assert np.isclose(expected_hamming_loss, measurements.hamming_loss_score())

    expected_jaccard_index = 0.6666666666666667
    assert np.isclose(expected_jaccard_index, measurements.jaccard_score())

    expected_info = ("WindowMultiLabelClassificationPerformanceEvaluator("
                     "n_labels=3, window_size=20, n_samples=20, "
                     "hamming_score=0.666667, hamming_loss_score=0.333333, "
                     "exact_match_score=0.250000, jaccard_score=0.666667)")
    assert expected_info == measurements.get_info()

    expected_last_true = (1.0, 1.0, 1.0)
    expected_last_pred = (1.0, 0.0, 1.0)
    assert np.alltrue(expected_last_true == measurements.get_last()[0])
    assert np.alltrue(expected_last_pred == measurements.get_last()[1])

    measurements.reset()
    assert measurements.n_samples == 0


def test_regression_measurements():
    y_true = np.sin(range(100))
    y_pred = np.sin(range(100)) + .05

    measurements = RegressionMeasurements()
    for i in range(len(y_true)):
        measurements.add_result(y_true[i], y_pred[i])

    expected_mse = 0.0025000000000000022
    assert np.isclose(expected_mse, measurements.get_mean_square_error())

    expected_ae = 0.049999999999999906
    assert np.isclose(expected_ae, measurements.get_average_error())

    expected_info = 'RegressionMeasurements: - sample_count: 100 - mean_square_error: 0.002500 ' \
                    '- mean_absolute_error: 0.050000'
    assert expected_info == measurements.get_info()

    expected_last = (-0.9992068341863537, -0.9492068341863537)
    assert np.alltrue(expected_last == measurements.get_last())

    measurements.reset()
    assert measurements.sample_count == 0


def test_window_regression_measurements():
    y_true = np.sin(range(100))
    y_pred = np.sin(range(100)) + .05

    measurements = WindowRegressionMeasurements(window_size=20)
    for i in range(len(y_true)):
        measurements.add_result(y_true[i], y_pred[i])

    expected_mse = 0.0025000000000000022
    assert np.isclose(expected_mse, measurements.get_mean_square_error())

    expected_ae = 0.050000000000000024
    assert np.isclose(expected_ae, measurements.get_average_error())

    expected_info = 'WindowRegressionMeasurements: - sample_count: 20 - mean_square_error: 0.002500 ' \
                    '- mean_absolute_error: 0.050000'
    assert expected_info == measurements.get_info()

    expected_last = (-0.9992068341863537, -0.9492068341863537)
    assert np.alltrue(expected_last == measurements.get_last())

    measurements.reset()
    assert measurements.sample_count == 0


def test_multi_target_regression_measurements():
    y_true = np.zeros((100, 3))
    y_pred = np.zeros((100, 3))

    for t in range(3):
        y_true[:, t] = np.sin(range(100))
        y_pred[:, t] = np.sin(range(100)) + (t + 1) * .05

    measurements = MultiTargetRegressionMeasurements()
    for i in range(100):
        measurements.add_result(y_true[i], y_pred[i])

    expected_amse = 0.011666666666666664
    assert np.isclose(expected_amse,
                      measurements.get_average_mean_square_error())

    expected_aae = 0.09999999999999999
    assert np.isclose(expected_aae, measurements.get_average_absolute_error())

    expected_armse = 0.09999999999999999
    assert np.isclose(expected_armse,
                      measurements.get_average_root_mean_square_error())

    expected_info = 'MultiTargetRegressionMeasurements: sample_count: 100 - ' \
                    'average_mean_square_error: {} - ' \
                    'average_mean_absolute_error: {} - ' \
                    'average_root_mean_square_error: {}'.format(
                        str(measurements.get_average_mean_square_error()),
                        str(measurements.get_average_absolute_error()),
                        str(measurements.get_average_root_mean_square_error())
                    )
    assert expected_info == measurements.get_info()

    expected_last = (np.array([-0.99920683, -0.99920683, -0.99920683]),
                     np.array([-0.94920683, -0.89920683, -0.84920683]))
    for exp, obs in zip(expected_last, measurements.get_last()):
        assert np.isclose(exp, obs).all()

    measurements.reset()
    assert measurements.sample_count == 0


def test_window_multi_target_regression_measurements():
    y_true = np.zeros((100, 3))
    y_pred = np.zeros((100, 3))

    for t in range(3):
        y_true[:, t] = np.sin(range(100))
        y_pred[:, t] = np.sin(range(100)) + (t + 1) * .05

    measurements = WindowMultiTargetRegressionMeasurements(window_size=20)
    for i in range(100):
        measurements.add_result(y_true[i], y_pred[i])

    expected_amse = 0.011666666666666672
    assert np.isclose(expected_amse,
                      measurements.get_average_mean_square_error())

    expected_aae = 0.10000000000000002
    assert np.isclose(expected_aae, measurements.get_average_absolute_error())

    expected_armse = 0.10000000000000003
    assert np.isclose(expected_armse,
                      measurements.get_average_root_mean_square_error())

    expected_info = 'MultiTargetRegressionMeasurements: sample_count: 20 - ' \
                    'average_mean_square_error: {} - ' \
                    'average_mean_absolute_error: {} - ' \
                    'average_root_mean_square_error: {}'.format(
                        str(measurements.get_average_mean_square_error()),
                        str(measurements.get_average_absolute_error()),
                        str(measurements.get_average_root_mean_square_error())
                    )
    assert expected_info == measurements.get_info()

    expected_last = (np.array([-0.99920683, -0.99920683, -0.99920683]),
                     np.array([-0.94920683, -0.89920683, -0.84920683]))
    for exp, obs in zip(expected_last, measurements.get_last()):
        assert np.isclose(exp, obs).all()

    measurements.reset()
    assert np.isclose(measurements.total_square_error, 0.0)


def test_running_time_measurements():
    rtm = RunningTimeMeasurements()

    for i in range(1000):
        # Test training time
        rtm.compute_training_time_begin()
        time.sleep(0.0005)
        rtm.compute_training_time_end()

        # Test testing time
        rtm.compute_testing_time_begin()
        time.sleep(0.0002)
        rtm.compute_testing_time_end()

        # Update statistics
        rtm.update_time_measurements()

    expected_info = 'RunningTimeMeasurements: sample_count: 1000 - ' \
                    'Total running time: {} - ' \
                    'training_time: {} - ' \
                    'testing_time: {}'.format(
                        rtm.get_current_total_running_time(),
                        rtm.get_current_training_time(),
                        rtm.get_current_testing_time(),
                    )

    assert expected_info == rtm.get_info()
