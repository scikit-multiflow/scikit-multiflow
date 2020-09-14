import os
import numpy as np
from skmultiflow.data.generator.stagger_generator import STAGGERGenerator


def test_stagger_generator(test_path):
    stream = STAGGERGenerator(classification_function=2, random_state=112, balance_classes=False)

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'stagger_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    for j in range(0,10):
        X, y = stream.next_sample()
        print(X_expected[j])
        print(X[0])
        print(y_expected[j])
        print(y[0])
        assert np.array_equal(X[0], X_expected[j])
        assert np.array_equal(y[0], y_expected[j])


    expected_info = "STAGGERGenerator(balance_classes=False, classification_function=2, random_state=112)"
    assert stream.get_info() == expected_info
