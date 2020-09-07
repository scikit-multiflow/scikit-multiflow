import os
import numpy as np
from skmultiflow.data.generator.led_generator_drift import LEDGeneratorDrift


def test_led_generator_drift(test_path):
    stream = LEDGeneratorDrift(random_state=112, noise_percentage=0.28, has_noise=True, n_drift_features=4)

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'led_stream_drift.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    for j in range(0,10):
        X, y = stream.next_sample()
        assert np.alltrue(np.isclose(X, X_expected[j]))
        assert np.alltrue(np.isclose(y[0], y_expected[j]))

    expected_info = "LEDGeneratorDrift(has_noise=True, n_drift_features=4, noise_percentage=0.28, random_state=112)"
    assert stream.get_info() == expected_info
