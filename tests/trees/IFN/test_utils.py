from skmultiflow.trees.ifn import utils
import pytest
import numpy as np


def test_suite_binary_search():
    arr = [1, 2, 3, 4, 5, 6]
    # exist value

    index = utils.binary_search(array=arr, left=0, right=5, value=3)
    assert index == 2
    # non exist value in the array
    index = utils.binary_search(array=arr, left=0, right=5, value=7)
    assert index == -1
    # non exist value between left and right
    index = utils.binary_search(array=arr, left=0, right=3, value=5)
    assert index == -1

    # invalid input
    index = utils.binary_search(array=None, left=0, right=3, value=5)
    assert index == -1
    index = utils.binary_search(array=arr, left=-1, right=3, value=5)
    assert index == -1
    index = utils.binary_search(array=arr, left=0, right=10, value=5)
    assert index == -1
    index = utils.binary_search(array=[], left=0, right=3, value=5)
    assert index == -1


def test_suite_split_data_to_two_intervals():
    interval = [(5, 0), (6, 0), (7, 1), (2, 1), (4, 0), (17, 1), (39, 0), (34, 0), (65, 1), (234, 1), (62, 0), (3, 1),
                (15, 1), (90, 0), (13, 1)]

    attribute_data, y = utils.split_data_to_two_intervals(interval=interval,
                                                          T=39,
                                                          min_value=13,
                                                          max_value=90)

    left_interval_size = attribute_data.count(0)
    assert left_interval_size == 4

    right_interval_size = attribute_data.count(1)
    assert right_interval_size == 4

    assert y == [1, 1, 1, 0, 0, 0, 1, 0]

    # invalid input
    interval = [1, 2, 3, 4, 5]

    with pytest.raises(TypeError):
        utils.split_data_to_two_intervals(interval=interval,
                                          T=39,
                                          min_value=13,
                                          max_value=90)


def test_suite_find_split_position():
    positions = [4, 8, 10]

    # value smaller than the smallest split point
    position = utils.find_split_position(value=1, positions=positions)
    assert position == 0
    # value equal to split point
    position = utils.find_split_position(value=4, positions=positions)
    assert position == 1
    # value between the first and second split point
    position = utils.find_split_position(value=5, positions=positions)
    assert position == 1
    # value equal to split point
    position = utils.find_split_position(value=8, positions=positions)
    assert position == 2
    # value between the second the third split point
    position = utils.find_split_position(value=9, positions=positions)
    assert position == 2
    # value bigger than the biggest split point
    position = utils.find_split_position(value=10, positions=positions)
    assert position == 3

    # invalid input
    with pytest.raises(IndexError):
        utils.find_split_position(value=11, positions=[])
    with pytest.raises(TypeError):
        utils.find_split_position(value=11, positions=None)


def test_suite_drop_records():
    X = [[1, 2, 3], [4, 2, 3], [3, 3, 3]]
    y = [0, 0, 1]
    expected_X = np.array([[1, 2, 3], [4, 2, 3]])
    expected_y = np.array([0, 0])

    new_X, new_y = utils.drop_records(X=X, y=y, attribute_index=1, value=2)

    assert np.array_equal(new_X, expected_X)
    assert np.array_equal(new_y, expected_y)

    new_X, new_y = utils.drop_records(X=X, y=y, attribute_index=1, value=4)

    assert np.array_equal(new_X, [])
    assert np.array_equal(new_y, [])

    # invalid input
    with pytest.raises(ValueError):
        utils.drop_records(X=[], y=y, attribute_index=1, value=4)
    with pytest.raises(ValueError):
        utils.drop_records(X=X, y=[0, 0], attribute_index=1, value=2)
    with pytest.raises(ValueError):
        utils.drop_records(X=[[1, 2, 3], [4, 2, 3]], y=[0, 0, 1], attribute_index=1, value=2)

    with pytest.raises(TypeError):
        utils.drop_records(X=[1, 2, 3], y=y, attribute_index=1, value=4)
    with pytest.raises(TypeError):
        utils.drop_records(X=None, y=y, attribute_index=1, value=4)
    with pytest.raises(TypeError):
        utils.drop_records(X=X, y=None, attribute_index=1, value=2)


def test_suite_create_attribute_node():
    X = [[1, 2, 3], [4, 2, 3], [3, 3, 3]]
    y = [0, 0, 1]
    chosen_attribute_index = 1
    attribute_value = 2
    curr_node_index = 1
    prev_node_index = 0

    attribute_node = utils.create_attribute_node(partial_X=X,
                                                 partial_y=y,
                                                 chosen_attribute_index=chosen_attribute_index,
                                                 attribute_value=attribute_value,
                                                 curr_node_index=curr_node_index,
                                                 prev_node_index=prev_node_index)

    assert attribute_node.prev_node == prev_node_index
    assert len(attribute_node.partial_x) == 2
    assert len(attribute_node.partial_y) == 2

    with pytest.raises(ValueError):
        utils.create_attribute_node(partial_X=X,
                                    partial_y=y,
                                    chosen_attribute_index=-1,
                                    attribute_value=attribute_value,
                                    curr_node_index=curr_node_index,
                                    prev_node_index=prev_node_index)
    with pytest.raises(ValueError):
        utils.create_attribute_node(partial_X=X,
                                    partial_y=y,
                                    chosen_attribute_index=chosen_attribute_index,
                                    attribute_value=-1,
                                    curr_node_index=curr_node_index,
                                    prev_node_index=prev_node_index)
    with pytest.raises(ValueError):
        utils.create_attribute_node(partial_X=X,
                                    partial_y=y,
                                    chosen_attribute_index=chosen_attribute_index,
                                    attribute_value=attribute_value,
                                    curr_node_index=-1,
                                    prev_node_index=prev_node_index)
    with pytest.raises(ValueError):
        utils.create_attribute_node(partial_X=X,
                                    partial_y=y,
                                    chosen_attribute_index=chosen_attribute_index,
                                    attribute_value=attribute_value,
                                    curr_node_index=curr_node_index,
                                    prev_node_index=-1)


def test_suite_convert_numeric_values():
    X = [[1, 2, 3], [4, 2, 3], [3, 3, 3]]
    chosen_split_points = [3]
    chosen_attribute = 0
    expected = [[0, 2, 3], [1, 2, 3], [1, 3, 3]]

    utils.convert_numeric_values(chosen_split_points=chosen_split_points,
                                 chosen_attribute=chosen_attribute,
                                 partial_X=X)

    assert np.array_equal(X, expected)

    X = []
    utils.convert_numeric_values(chosen_split_points=chosen_split_points,
                                 chosen_attribute=chosen_attribute,
                                 partial_X=[])

    assert np.array_equal(X, [])

    X = [[1, 2, 3], [4, 2, 3], [3, 3, 3]]

    # invalid input
    with pytest.raises(IndexError):
        utils.convert_numeric_values(chosen_split_points=chosen_split_points, chosen_attribute=3, partial_X=X)
    with pytest.raises(AttributeError):
        utils.convert_numeric_values(chosen_split_points=None, chosen_attribute=chosen_attribute, partial_X=X)
    with pytest.raises(TypeError):
        utils.convert_numeric_values(chosen_split_points=chosen_split_points, chosen_attribute=chosen_attribute,
                                     partial_X=None)


def test_suite_calculate_second_best_attribute_of_last_layer():
    mi_dict = {0: 0, 1: 0, 2: 0.829, 3: 0.527}

    index, mi = utils.calculate_second_best_attribute_of_last_layer(attributes_mi=mi_dict)
    assert index == 3
    assert mi == 0.527

    mi_dict = {0: 0, 1: 0, 2: 0.829, 3: 0}

    index, mi = utils.calculate_second_best_attribute_of_last_layer(attributes_mi=mi_dict)
    assert index == -1
    assert mi == 0

    index, mi = utils.calculate_second_best_attribute_of_last_layer(attributes_mi=None)
    assert index == -1
    assert mi == 0

    index, mi = utils.calculate_second_best_attribute_of_last_layer(attributes_mi={2: 0.829})
    assert index == -1
    assert mi == 0

    index, mi = utils.calculate_second_best_attribute_of_last_layer(attributes_mi={})
    assert index == -1
    assert mi == 0


test_suite_binary_search()
test_suite_split_data_to_two_intervals()
test_suite_find_split_position()
test_suite_drop_records()
test_suite_create_attribute_node()
test_suite_convert_numeric_values()
test_suite_calculate_second_best_attribute_of_last_layer()
