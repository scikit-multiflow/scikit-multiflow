import os
import numpy as np
import pandas as pd
from skmultiflow.data import DataGenerator


def test_data_stream_X_y(test_path, package_path):
    test_file = os.path.join(package_path, 'src/skmultiflow/data/datasets/sea_stream.csv')
    raw_data = pd.read_csv(test_file)
    y = raw_data.iloc[:, -1:]
    X = raw_data.iloc[:, :-1]
    X_gen, y_gen = DataGenerator(X), DataGenerator(y)

    assert not y_gen.data.empty

    X_gen.prepare_for_use()
    y_gen.prepare_for_use()

    assert X_gen.n_remaining_samples() == 40000
    assert y_gen.n_remaining_samples() == 40000

    expected_names = ['attrib1', 'attrib2', 'attrib3']
    assert X_gen.data.columns.to_list() == expected_names

    expected_targets = {0, 1}
    assert set(y_gen.data['class'].unique()) == expected_targets

    assert y_gen.data.columns.to_list() == ['class']

    assert X_gen.data.shape[1] == 3

    assert y_gen.data.shape[1] == 1

    assert X_gen.has_more_samples() is True
    assert y_gen.has_more_samples() is True

    assert X_gen.is_restartable() is True
    assert y_gen.is_restartable() is True


    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'sea_stream_file.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    X, y = X_gen.next_sample(), y_gen.next_sample()
    assert np.alltrue(X.iloc[0, :] == X_expected[0])
    assert np.alltrue(y.iloc[0, :] == y_expected[0])

    X_gen.restart(), y_gen.restart()
    X, y = X_gen.next_sample(10), y_gen.next_sample(10)
    assert np.alltrue(X.values == X_expected)
    assert np.alltrue(y.values.reshape(-1) == y_expected)
