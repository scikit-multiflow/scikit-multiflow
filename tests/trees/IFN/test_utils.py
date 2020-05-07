from skmultiflow.trees.ifn import utils
import pytest
import numpy as np


def test_suite_binary_search():
    arr = [1, 2, 3, 4, 5, 6]
    # exist value
    test_binary_search(arr=arr, left=0, right=5, value=3, expected=2)
    # non exist value in the array
    test_binary_search(arr=arr, left=0, right=5, value=7, expected=-1)
    # non exist value between left and right
    test_binary_search(arr=arr, left=0, right=3, value=5, expected=-1)
    # invalid input
    test_binary_search(arr=None, left=0, right=3, value=5, expected=-1)
    test_binary_search(arr=arr, left=-1, right=3, value=5, expected=-1)
    test_binary_search(arr=arr, left=0, right=10, value=5, expected=-1)
    test_binary_search(arr=[], left=0, right=3, value=5, expected=-1)


def test_binary_search(arr, left, right, value, expected):
    index = utils.binary_search(array=arr, left=left, right=right, value=value)

    assert index == expected


def test_suite_split_data_to_two_intervals():
    interval = [(5, 0), (6, 0), (7, 1), (2, 1), (4, 0), (17, 1), (39, 0), (34, 0), (65, 1), (234, 1), (62, 0), (3, 1),
                (15, 1), (90, 0), (13, 1)]

    test_split_data_to_two_intervals(interval=interval, T=39, min_value=13, max_value=90, expected_left_size=4,
                                     expected_right_size=4, expected_y_values=[1, 1, 1, 0, 0, 0, 1, 0])

    # invalid input
    interval = [1, 2, 3, 4, 5]

    with pytest.raises(TypeError):
        test_split_data_to_two_intervals(interval=interval, T=39, min_value=13, max_value=90)


def test_split_data_to_two_intervals(interval, T, min_value, max_value, expected_left_size=None,
                                     expected_right_size=None,
                                     expected_y_values=None):
    attribute_data, y = utils.split_data_to_two_intervals(interval=interval,
                                                          T=T,
                                                          min_value=min_value,
                                                          max_value=max_value)

    left_interval_size = attribute_data.count(0)
    assert left_interval_size == expected_left_size

    right_interval_size = attribute_data.count(1)
    assert right_interval_size == expected_right_size

    assert y == expected_y_values


def test_suite_find_split_position():
    positions = [4, 8, 10]

    # value smaller than the smallest split point
    test_find_split_position(positions=positions, value=1, expected=0)
    # value equal to split point
    test_find_split_position(positions=positions, value=4, expected=1)
    # value between the first and second split point
    test_find_split_position(positions=positions, value=5, expected=1)
    # value equal to split point
    test_find_split_position(positions=positions, value=8, expected=2)
    # value between the second the third split point
    test_find_split_position(positions=positions, value=9, expected=2)
    # value bigger than the biggest split point
    test_find_split_position(positions=positions, value=11, expected=3)

    # invalid input
    with pytest.raises(IndexError):
        test_find_split_position(positions=[], value=11)
    with pytest.raises(TypeError):
        test_find_split_position(positions=None, value=11)


def test_find_split_position(positions, value, expected=None):
    position = utils.find_split_position(value=value, positions=positions)
    assert position == expected


def test_suite_drop_records():
    X = [[1, 2, 3], [4, 2, 3], [3, 3, 3]]
    y = [0, 0, 1]
    expected_X = np.array([[1, 2, 3], [4, 2, 3]])
    expected_y = np.array([0, 0])

    test_drop_records(X=X, y=y, index=1, value=2, expected_X=expected_X, expected_y=expected_y)
    test_drop_records(X=X, y=y, index=1, value=4, expected_X=[], expected_y=[])

    # invalid input
    with pytest.raises(ValueError):
        test_drop_records(X=[], y=y, index=1, value=4)
        test_drop_records(X=X, y=[0, 0], index=1, value=2)
        test_drop_records(X=[[1, 2, 3], [4, 2, 3]], y=[0, 0, 1], index=1, value=2)
    with pytest.raises(TypeError):
        test_drop_records(X=[1, 2, 3], y=y, index=1, value=4)
        test_drop_records(X=None, y=y, index=1, value=4)
        test_drop_records(X=X, y=None, index=1, value=2)


def test_drop_records(X, y, index, value, expected_X=None, expected_y=None):
    new_X, new_y = utils.drop_records(X=X, y=y, attribute_index=index, value=value)

    assert np.array_equal(new_X, expected_X)
    assert np.array_equal(new_y, expected_y)


def test_suite_create_attribute_node():
    X = [[1, 2, 3], [4, 2, 3], [3, 3, 3]]
    y = [0, 0, 1]
    chosen_attribute_index = 1
    attribute_value = 2
    curr_node_index = 1
    prev_node_index = 0

    test_create_attribute_node(X=X,
                               y=y,
                               chosen_attribute_index=chosen_attribute_index,
                               attribute_value=attribute_value,
                               curr_node_index=curr_node_index,
                               prev_node_index=prev_node_index,
                               expected_X_length=2,
                               expected_y_length=2)

    with pytest.raises(ValueError):
        test_create_attribute_node(X=X,
                                   y=y,
                                   chosen_attribute_index=-1,
                                   attribute_value=attribute_value,
                                   curr_node_index=curr_node_index,
                                   prev_node_index=prev_node_index)
        test_create_attribute_node(X=X,
                                   y=y,
                                   chosen_attribute_index=chosen_attribute_index,
                                   attribute_value=-1,
                                   curr_node_index=curr_node_index,
                                   prev_node_index=prev_node_index)
        test_create_attribute_node(X=X,
                                   y=y,
                                   chosen_attribute_index=chosen_attribute_index,
                                   attribute_value=attribute_value,
                                   curr_node_index=-1,
                                   prev_node_index=prev_node_index)

        test_create_attribute_node(X=X,
                                   y=y,
                                   chosen_attribute_index=chosen_attribute_index,
                                   attribute_value=attribute_value,
                                   curr_node_index=curr_node_index,
                                   prev_node_index=-1)


def test_create_attribute_node(X, y, chosen_attribute_index, attribute_value, curr_node_index, prev_node_index,
                               expected_X_length=None, expected_y_length=None):
    attribute_node = utils.create_attribute_node(partial_X=X,
                                                 partial_y=y,
                                                 chosen_attribute_index=chosen_attribute_index,
                                                 attribute_value=attribute_value,
                                                 curr_node_index=curr_node_index,
                                                 prev_node_index=prev_node_index)

    assert attribute_node.prev_node == prev_node_index
    assert len(attribute_node.partial_x) == expected_X_length
    assert len(attribute_node.partial_y) == expected_y_length


def test_suite_convert_numeric_values():
    X = [[1, 2, 3], [4, 2, 3], [3, 3, 3]]
    chosen_split_points = [3]
    chosen_attribute = 0
    expected = [[0, 2, 3], [1, 2, 3], [1, 3, 3]]

    test_convert_numeric_values(chosen_split_points=chosen_split_points, chosen_attribute=chosen_attribute, X=X,
                                expected=expected)
    test_convert_numeric_values(chosen_split_points=chosen_split_points, chosen_attribute=chosen_attribute, X=[],
                                expected=[])

    # invalid input
    with pytest.raises(IndexError):
        test_convert_numeric_values(chosen_split_points=chosen_split_points, chosen_attribute=3, X=X)
    with pytest.raises(AttributeError):
        test_convert_numeric_values(chosen_split_points=None, chosen_attribute=chosen_attribute, X=X)
    with pytest.raises(TypeError):
        test_convert_numeric_values(chosen_split_points=chosen_split_points, chosen_attribute=chosen_attribute, X=None)


def test_convert_numeric_values(chosen_split_points, chosen_attribute, X, expected=None):
    utils.convert_numeric_values(chosen_split_points=chosen_split_points,
                                 chosen_attribute=chosen_attribute,
                                 partial_X=X)

    assert np.array_equal(X, expected)


def test_suite_calculate_second_best_attribute_of_last_layer():
    mi_dict = {0: 0, 1: 0, 2: 0.829, 3: 0.527}

    test_calculate_second_best_attribute_of_last_layer(mi_dict=mi_dict, expected_index=3, expected_mi=0.527)

    mi_dict = {0: 0, 1: 0, 2: 0.829, 3: 0}
    test_calculate_second_best_attribute_of_last_layer(mi_dict=mi_dict, expected_index=-1, expected_mi=0)

    with pytest.raises(AttributeError):
        test_calculate_second_best_attribute_of_last_layer(mi_dict=None)
    with pytest.raises(ValueError):
        test_calculate_second_best_attribute_of_last_layer(mi_dict={})


def test_calculate_second_best_attribute_of_last_layer(mi_dict, expected_index=None, expected_mi=None):
    index, mi = utils.calculate_second_best_attribute_of_last_layer(attributes_mi=mi_dict)

    assert index == expected_index
    assert mi == expected_mi


test_suite_binary_search()
test_suite_split_data_to_two_intervals()
test_suite_find_split_position()
test_suite_drop_records()
test_suite_create_attribute_node()
test_suite_convert_numeric_values()
test_suite_calculate_second_best_attribute_of_last_layer()
