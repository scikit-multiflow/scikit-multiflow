import os
import numpy as np
from skmultiflow.data.generator.led_generator import LEDGenerator


def test_led_generator(test_path):
    stream = LEDGenerator(random_state=112, noise_percentage=0.28, has_noise=True)

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'led_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    for j in range(0,10):
        X, y = stream.next_sample()
        assert np.alltrue(np.isclose(X, X_expected[j]))
        assert np.alltrue(np.isclose(y[0], y_expected[j]))

    expected_info = "LEDGenerator(has_noise=True, noise_percentage=0.28, random_state=112)"
    assert stream.get_info() == expected_info
