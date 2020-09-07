import os
import numpy as np
from skmultiflow.data.generator.multilabel_generator import MultilabelGenerator


def test_multilabel_generator(test_path):
    stream = MultilabelGenerator(n_samples=100, n_features=20, n_targets=4, n_labels=4, random_state=0)

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'multilabel_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    for j in range(0,10):
        X, y = stream.next_sample()
        assert np.alltrue(np.isclose(X, X_expected[j]))
        assert np.alltrue(np.isclose(y[0], y_expected[j]))

    expected_info = "MultilabelGenerator(n_features=20, n_labels=4, n_samples=100, n_targets=4, random_state=0)"
    assert stream.get_info() == expected_info
