import os
import numpy as np
from skmultiflow.data.generator.waveform_generator import WaveformGenerator


def test_waveform_generator(test_path):
    stream = WaveformGenerator(random_state=23, has_noise=False)

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'waveform_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    for j in range(0,10):
        X, y = stream.next_sample()
        assert np.array_equal(X[0], X_expected[j])
        assert np.array_equal(y[0], y_expected[j])


    expected_info = "WaveformGenerator(has_noise=False, random_state=23)"
    assert stream.get_info() == expected_info

def test_waveform_generator_noise(test_path):
    stream = WaveformGenerator(random_state=23, has_noise=True)

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'waveform_noise_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    for j in range(0,10):
        X, y = stream.next_sample()
        assert np.array_equal(X[0], X_expected[j])
        assert np.array_equal(y[0], y_expected[j])


    expected_info = "WaveformGenerator(has_noise=True, random_state=23)"
    assert stream.get_info() == expected_info
