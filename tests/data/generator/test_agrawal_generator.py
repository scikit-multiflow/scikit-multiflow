import os
import numpy as np
from skmultiflow.data.generator.agrawal_generator import AGRAWALGenerator


def test_agrawal_generator(test_path):
    stream = AGRAWALGenerator(classification_function=2, random_state=112, balance_classes=False, perturbation=0.28)

    assert stream.name == 'AGRAWAL Generator'

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'agrawal_stream.npz')
    data = np.load(test_file)

    X_expected = data['X']
    y_expected = data['y']

    X, y = stream.next_sample()
    assert np.alltrue(X[0] == X_expected[0])
    assert np.alltrue(y[0] == y_expected[0])

    expected_info = "AGRAWALGenerator(balance_classes=False, classification_function=2, perturbation=0.28, random_state=112)"
    assert stream.get_info() == expected_info


def test_agrawal_generator_all_functions(test_path):
    for f in range(10):
        stream = AGRAWALGenerator(classification_function=f, random_state=1)

        # Load test data corresponding to first 10 instances
        test_file = os.path.join(test_path, 'agrawal_stream_{}.npz'.format(f))
        data = np.load(test_file)
        X_expected = data['X']
        y_expected = data['y']

        for j in range(0,10):
            X, y = stream.next_sample()
            assert np.alltrue(X == X_expected[j])
            assert np.alltrue(y == y_expected[j])


def test_agrawal_drift(test_path):
    stream = AGRAWALGenerator(random_state=1)

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'agrawal_stream_drift.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    for j in range(0,10):
        X, y = stream.next_sample()
        assert np.alltrue(X == X_expected[j])
        assert np.alltrue(y == y_expected[j])

    stream.generate_drift()

    for j in range(10,20):
        X_drift, y_drift = stream.next_sample()
        assert np.alltrue(X_drift == X_expected[j])
        assert np.alltrue(y_drift == y_expected[j])
