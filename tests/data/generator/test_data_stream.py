import os
import numpy as np
import pandas as pd

import pytest   # noqa

from skmultiflow.data.data_stream import DataStream


def test_data_stream(test_path):
    test_file = os.path.join(test_path, 'sea_stream_file.csv')
    raw_data = pd.read_csv(test_file)
    stream = DataStream(raw_data, name='Test')

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'sea_stream_file.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    for j in range(0,10):
        X, y = stream.next_sample()
        assert np.alltrue(X == X_expected[j])
        assert np.alltrue(y == y_expected[j])

    expected_info = "DataStream(n_targets=-1, target_idx=1, cat_features=None, name='Test')"
    assert stream.get_info() == expected_info


def test_data_stream_X_y(test_path):
    test_file = os.path.join(test_path, 'sea_stream_file.csv')
    raw_data = pd.read_csv(test_file)
    y = raw_data.iloc[:, -1:]
    X = raw_data.iloc[:, :-1]
    stream = DataStream(X, y)

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'sea_stream_file.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    for j in range(0,10):
        X, y = stream.next_sample()
        assert np.alltrue(X == X_expected[j])
        assert np.alltrue(y == y_expected[j])

    # Ensure that the regression case is also covered
    y = raw_data.iloc[:, -1:]
    X = raw_data.iloc[:, :-1]
    y = y.astype('float64')
    stream = DataStream(X, y, name='Test')



def test_check_data():
    # Test if data contains non-numeric values
    data = pd.DataFrame(np.array([[1, 2, 3, 4, 5],
                                  [6, 7, 8, 9, 10],
                                  [11, 'invalid', 13, 14, 15]]))

    with pytest.raises(ValueError):
        DataStream(data=data, allow_nan=False)

    # Test if data contains NaN values
    data = pd.DataFrame(np.array([[1, 2, 3, 4, 5],
                                  [6, 7, 8, 9, 10],
                                  [11, np.nan, 13, 14, 15]]))

    with pytest.raises(ValueError):
        DataStream(data=data, allow_nan=False)

    # Test warning for NaN values

    with pytest.warns(UserWarning):
        DataStream(data=data, allow_nan=True)
