import os
import numpy as np
from skmultiflow.data.generator.hyperplane_generator import HyperplaneGenerator


def test_hyper_plane_generator(test_path):
    n_features = 10
    stream = HyperplaneGenerator(random_state=112, n_features=n_features, n_drift_features=2, mag_change=0.6,
                                 noise_percentage=0.28, sigma_percentage=0.1)

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'hyper_plane_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    for j in range(0,10):
        X, y = stream.next_sample()
        assert np.alltrue(np.isclose(X, X_expected[j]))
        assert np.alltrue(np.isclose(y[0], y_expected[j]))

    expected_info = "HyperplaneGenerator(mag_change=0.6, n_drift_features=2, n_features=10, noise_percentage=0.28, random_state=112, sigma_percentage=0.1)"
    assert stream.get_info() == expected_info
