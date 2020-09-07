import os
import numpy as np
from skmultiflow.data.generator.random_rbf_generator import RandomRBFGenerator


def test_random_rbf_generator(test_path):
    stream = RandomRBFGenerator(model_random_state=99, sample_random_state=50, n_classes=4, n_features=10, n_centroids=50)

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'random_rbf_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    for j in range(0,10):
        X, y = stream.next_sample()
        assert np.alltrue(np.isclose(X, X_expected[j]))
        assert np.alltrue(np.isclose(y[0], y_expected[j]))

    expected_info = "RandomRBFGenerator(model_random_state=99, n_centroids=50, n_classes=4, n_features=10, sample_random_state=50)"

    assert stream.get_info() == expected_info
