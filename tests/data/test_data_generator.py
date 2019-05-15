import os
import numpy as np
import pandas as pd
from skmultiflow.data import DataGenerator


def test_data_stream_X(test_path, package_path):
    test_file = os.path.join(package_path, 'src/skmultiflow/data/datasets/sea_stream.csv')
    raw_data = pd.read_csv(test_file)
    X = raw_data.iloc[:, :-1]
    X_gen = DataGenerator(X)

    X_gen.prepare_for_use()

    assert X_gen.n_remaining_samples() == 40000

    expected_names = ['attrib1', 'attrib2', 'attrib3']
    assert X_gen.data.columns.to_list() == expected_names

    assert X_gen.data.shape[1] == 3

    assert X_gen.has_more_samples() is True

    assert X_gen.is_restartable() is True

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'sea_stream_file.npz')
    data = np.load(test_file)
    X_expected = data['X']

    X = X_gen.next_sample()
    assert np.alltrue(X.iloc[0, :] == X_expected[0])

    X_gen.restart()
    X = X_gen.next_sample(10)
    assert np.alltrue(X.values == X_expected)


def test_data_stream_y(test_path, package_path):
    test_file = os.path.join(package_path, 'src/skmultiflow/data/datasets/sea_stream.csv')
    raw_data = pd.read_csv(test_file)
    y = raw_data.iloc[:, -1:]
    y_gen = DataGenerator(y)

    assert not y_gen.data.empty
    y_gen.prepare_for_use()

    assert y_gen.n_remaining_samples() == 40000

    expected_targets = {0, 1}
    assert set(y_gen.data['class'].unique()) == expected_targets

    assert y_gen.data.columns.to_list() == ['class']

    assert y_gen.data.shape[1] == 1
    assert y_gen.has_more_samples() is True
    assert y_gen.is_restartable() is True


    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'sea_stream_file.npz')
    data = np.load(test_file)
    y_expected = data['y']

    y = y_gen.next_sample()
    assert np.alltrue(y.iloc[0, :] == y_expected[0])

    y_gen.restart()
    y = y_gen.next_sample(10)
    assert np.alltrue(y.values.reshape(-1) == y_expected)
