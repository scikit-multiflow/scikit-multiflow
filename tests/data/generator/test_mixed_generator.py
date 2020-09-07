import os
import numpy as np
from skmultiflow.data.generator.mixed_generator import MIXEDGenerator


def test_mixed_generator(test_path):
    stream = MIXEDGenerator(classification_function=1, random_state=112, balance_classes=False)

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'mixed_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    for j in range(0,10):
        X, y = stream.next_sample()
        assert np.alltrue(np.isclose(X, X_expected[j]))
        assert np.alltrue(np.isclose(y[0], y_expected[j]))

    expected_info = "MIXEDGenerator(balance_classes=False, classification_function=1, random_state=112)"
    assert stream.get_info() == expected_info
