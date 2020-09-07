import os

import pytest

import numpy as np

from skmultiflow.data.generator.anomaly_sine_generator import AnomalySineGenerator


def test_anomaly_sine_generator(test_path):
    stream = AnomalySineGenerator(random_state=12345,
                                  n_samples=100,
                                  n_anomalies=25,
                                  contextual=True,
                                  n_contextual=10)

    # Load test data corresponding to first 100 instances
    test_file = os.path.join(test_path, 'anomaly_sine_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    for j in range(0,100):
        X, y = stream.next_sample()
        assert np.alltrue(np.isclose(X, X_expected[j].reshape((1, 2))))
        assert np.alltrue(np.isclose(y[0], y_expected[j]))

    expected_info = "AnomalySineGenerator(contextual=True, n_anomalies=25, n_contextual=10, " \
                    "n_samples=100, noise=0.5, random_state=12345, replace=True, shift=4)"
    info = " ".join([line.strip() for line in stream.get_info().split()])
    assert info == expected_info

    # Coverage
    with pytest.raises(ValueError):
        # Invalid n_anomalies
        AnomalySineGenerator(n_samples=100, n_anomalies=250)

    with pytest.raises(ValueError):
        # Invalid n_contextual
        AnomalySineGenerator(n_samples=100, n_anomalies=50, contextual=True, n_contextual=250)
