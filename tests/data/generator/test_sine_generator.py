import os
import numpy as np
from skmultiflow.data.generator.sine_generator import SineGenerator


def test_sine_generator(test_path):
    stream = SineGenerator(classification_function=2, random_state=112, balance_classes=False, has_noise=True)

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'sine_noise_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    for j in range(0,10):
        X, y = stream.next_sample()
        assert np.alltrue(np.isclose(X, X_expected[j]))
        assert np.alltrue(np.isclose(y[0], y_expected[j]))


    expected_info = "SineGenerator(balance_classes=False, classification_function=2, has_noise=True, random_state=112)"
    assert stream.get_info() == expected_info
