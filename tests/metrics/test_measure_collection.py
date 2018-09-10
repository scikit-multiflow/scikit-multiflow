import numpy as np
from skmultiflow.metrics import ClassificationMeasurements
from skmultiflow.metrics import WindowClassificationMeasurements
from skmultiflow.metrics import MultiOutputMeasurements
from skmultiflow.metrics import WindowMultiOutputMeasurements
from skmultiflow.metrics import RegressionMeasurements
from skmultiflow.metrics import WindowRegressionMeasurements


def test_classification_measurements():
    y_true = np.concatenate((np.ones(85), np.zeros(10), np.ones(5)))
    y_pred = np.concatenate((np.ones(90), np.zeros(10)))

    measurements = ClassificationMeasurements()
    for i in range(len(y_true)):
        measurements.add_result(y_true[i], y_pred[i])

    expected_acc = .9
    assert expected_acc == measurements.get_performance()

    expected_incorrectly_classified_ratio = 1 - expected_acc
    assert expected_incorrectly_classified_ratio == measurements.get_incorrectly_classified_ratio()

    expected_kappa = (expected_acc - 0.82) / (1 - 0.82)
    assert np.isclose(expected_kappa, measurements.get_kappa())

    expected_kappa_m = (expected_acc - .1) / (1 - 0.1)
    assert np.isclose(expected_kappa_m, measurements.get_kappa_m())

    expected_kappa_t = (expected_acc - .97) / (1 - 0.97)
    assert expected_kappa_t == measurements.get_kappa_t()

    expected_info = 'ClassificationMeasurements: - sample_count: 100.0 - performance: 0.900000 - kappa: 0.444444 ' \
                    '- kappa_t: -2.333333 - kappa_m: 0.888889 - majority_class: 0'
    assert expected_info == measurements.get_info()

    expected_last = (1.0, 0.0)
    assert expected_last == measurements.get_last()

    expected_majority_class = 0
    assert expected_majority_class == measurements.get_majority_class()


def test_window_classification_measurements():
    y_true = np.ones(100)
    y_pred = np.concatenate((np.ones(90), np.zeros(10)))

    measurements = WindowClassificationMeasurements(window_size=20)
    for i in range(len(y_true)):
        measurements.add_result(y_true[i], y_pred[i])

    expected_acc = .5
    assert expected_acc == measurements.get_performance()

    expected_incorrectly_classified_ratio = 1 - expected_acc
    assert expected_incorrectly_classified_ratio == measurements.get_incorrectly_classified_ratio()

    expected_kappa = 0.0
    assert expected_kappa == measurements.get_kappa()

    expected_kappa_m = 0.5
    assert expected_kappa_m == measurements.get_kappa_m()

    expected_kappa_t = 1
    assert expected_kappa_t == measurements.get_kappa_t()

    expected_info = 'WindowClassificationMeasurements: - sample_count: 20 - window_size: 20 - performance: 0.500000 ' \
                    '- kappa: 0.000000 - kappa_t: 1.000000 - kappa_m: 0.500000 - majority_class: 0'
    assert expected_info == measurements.get_info()

    expected_last = (1.0, 0.0)
    assert expected_last == measurements.get_last()

    expected_majority_class = 0
    assert expected_majority_class == measurements.get_majority_class()


def test_multi_output_measurements():
    y_0 = np.ones(100)
    y_1 = np.concatenate((np.ones(90), np.zeros(10)))
    y_2 = np.concatenate((np.ones(85), np.zeros(10), np.ones(5)))
    y_true = np.ones((100, 3))
    y_pred = np.vstack((y_0, y_1, y_2)).T

    measurements = MultiOutputMeasurements()
    for i in range(len(y_true)):
        measurements.add_result(y_true[i], y_pred[i])

    expected_acc = 0.85
    assert np.isclose(expected_acc, measurements.get_exact_match())

    expected_hamming_score = 1 - 0.06666666666666667
    assert np.isclose(expected_hamming_score, measurements.get_hamming_score())

    expected_hamming_loss = 0.06666666666666667
    assert np.isclose(expected_hamming_loss, measurements.get_hamming_loss())

    expected_jaccard_index = 0.9333333333333332
    assert np.isclose(expected_jaccard_index, measurements.get_j_index())

    expected_total_sum = 300
    assert expected_total_sum == measurements.get_total_sum()

    expected_info = 'MultiOutputMeasurements: - sample_count: 100 - hamming_loss: 0.066667 - hamming_score: 0.933333 ' \
                    '- exact_match: 0.850000 - j_index: 0.933333'
    assert expected_info == measurements.get_info()

    expected_last_true = (1.0, 1.0, 1.0)
    expected_last_pred = (1.0, 0.0, 1.0)
    assert np.alltrue(expected_last_true == measurements.get_last()[0])
    assert np.alltrue(expected_last_pred == measurements.get_last()[1])


def test_window_multi_output_measurements():
    y_0 = np.ones(100)
    y_1 = np.concatenate((np.ones(90), np.zeros(10)))
    y_2 = np.concatenate((np.ones(85), np.zeros(10), np.ones(5)))
    y_true = np.ones((100, 3))
    y_pred = np.vstack((y_0, y_1, y_2)).T

    measurements = WindowMultiOutputMeasurements(window_size=20)
    for i in range(len(y_true)):
        measurements.add_result(y_true[i], y_pred[i])

    expected_acc = 0.25
    assert np.isclose(expected_acc, measurements.get_exact_match())

    expected_hamming_score = 1 - 0.33333333333333337
    assert np.isclose(expected_hamming_score, measurements.get_hamming_score())

    expected_hamming_loss = 0.33333333333333337
    assert np.isclose(expected_hamming_loss, measurements.get_hamming_loss())

    expected_jaccard_index = 0.6666666666666667
    assert np.isclose(expected_jaccard_index, measurements.get_j_index())

    expected_total_sum = 300
    assert expected_total_sum == measurements.get_total_sum()

    expected_info = 'WindowMultiOutputMeasurements: - sample_count: 20 - hamming_loss: 0.333333 ' \
                    '- hamming_score: 0.666667 - exact_match: 0.250000 - j_index: 0.666667'
    assert expected_info == measurements.get_info()

    expected_last_true = (1.0, 1.0, 1.0)
    expected_last_pred = (1.0, 0.0, 1.0)
    assert np.alltrue(expected_last_true == measurements.get_last()[0])
    assert np.alltrue(expected_last_pred == measurements.get_last()[1])


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
