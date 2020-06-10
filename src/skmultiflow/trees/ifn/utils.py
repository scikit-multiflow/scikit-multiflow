import numpy as np
from .ifn_network import AttributeNode


def binary_search(array: list, left: int, right: int, value):
    """

    Parameters
    ----------
    array: (An array_like object of length n)
        A sorted array of int/float.

    left: int
        Index in the array.

    right: int
        Index in the array.

    value: int/float
        The value needed to be founded in the array.


    Returns
    -------
        The index of the value in the array or -1 if something went wrong


    """
    if array is None or len(array) == 0 or left < 0 or len(array) < right:
        return -1

    # Check base case
    if right >= left:
        mid = int(left + (right - left) / 2)
        # If element is present at the middle itself
        if (mid == 0 or value > array[mid - 1]) and array[mid] == value:
            return mid
        # If element is smaller than mid, then it can only
        # be present in left subarray
        # elif array[mid] > value:
        #     return binary_search(array, left, mid - 1, value)
        # # Else the element can only be present in right subarray
        # else:
        #     return binary_search(array, mid + 1, right, value)
        elif value > array[mid]:
            return binary_search(array, mid + 1, right, value)
        else:
            return binary_search(array, left, mid - 1,value)
    else:
        # Element is not present in the array
        return -1


def split_data_to_two_intervals(interval, T, min_value, max_value):
    """ Splitting the given interval to two intervals contains only the number between min_value and max_value.
        One interval contains all the data which is smaller than T (will be referred as left interval).
        while the second interval contain all the data which is equal or larger than T (will be referred as right
        interval)

    Parameters
    ----------
    interval: {array-like, sparse matrix}, shape (1_sample, n_classes)
        Contains the data of one feature overall samples in the train set.

    T: int or float
        Threshold point

    min_value: int or float
        minimum value in the interval

    max_value: int or float
        maximum value in the interval

    Returns
    -------
        An array_like object of length n_samples contains 0's equal to the number of values which are smaller than
        T and 1's equal to the number of values which are equal of larger than T.
        An array_like object of length n_samples which contains the true class labels for all the samples in the
        interval.
    """
    if interval is None:
        raise ValueError("interval shouldn't be None")

    t_attribute_data = []
    new_y = []

    interval.sort(key=lambda tup: tup[0])

    for data_class_tuple in interval:
        if min_value <= data_class_tuple[0] <= max_value:
            new_y.append(data_class_tuple[1])
            if data_class_tuple[0] < T:
                t_attribute_data.append(0)
            else:
                t_attribute_data.append(1)

    return t_attribute_data, new_y


def find_split_position(value, positions: list):
    """ Find the position of the given value between the list of given positions.

    Parameters
    ----------
    value: int or float
        A value of the chosen attribute upon the current layer in the network will be splited by.

    positions: list
        List of the chosen split points which founded significant for the chosen attribute upon the current
        layer in the network will be splited by.


    Returns
    -------
        The position after discretization of the given value among the positions list.

    """

    # If value is smaller than the first split point
    if value < positions[0]:
        return 0
    # If value is equal/larger than the first split point
    if positions[len(positions) - 1] <= value:
        return len(positions)

    for i in range(len(positions)):
        first_position = positions[i]
        second_position = positions[i + 1]
        if first_position <= value < second_position:
            return i + 1


def write_details_to_file(layer_position, attributes_cmi, chosen_attribute_index, chosen_attribute):
    """ Write network details to a file name 'output.txt'.

    Parameters
    ----------
    layer_position: string
        current layer position in the network - first/next.

    attributes_cmi: (dictionary) {attribute_index : conditional mutual information}
        Contains the conditional mutual information of each attribute.

    chosen_attribute_index: int or float
        The chosen attribute index upon the network will be splited by.

    chosen_attribute: string
        The chosen attribute name upon the network will be splited by.



    """
    with open('output.txt', 'a') as f:
        f.write(layer_position + ' layer attribute: \n')
        for index, mi in attributes_cmi.items():
            f.write(str(index) + ': ' + str(round(mi, 3)) + '\n')

        if chosen_attribute_index != -1:
            f.write('\nChosen attribute is: ' + chosen_attribute + "(" + str(chosen_attribute_index) + ")" + '\n\n')
        else:
            f.write('\nChosen attribute is: None' + "(" + str(chosen_attribute_index) + ")" + '\n\n')
        f.close()


def drop_records(X, y, attribute_index, value):
    """ Drop the samples in X which doesn't equal to value.

    Parameters
    ----------
    x: {array-like, sparse matrix}, shape (n_samples, n_features)
        The samples from a specific node.

    y: (An array_like object of length n_samples)
        Contains the true class labels for all the samples in X.

    attribute_index: int
        The index of the attribute upon the network will be splited by.

    value: int or float
        The data value to keep in x.


    Returns
    -------
        np.array of the samples which their data value in attribute_index is equal to value.

    """
    new_x = []
    new_y = []

    if len(X) != len(y):
        raise ValueError("X and y should have the same number of rows")

    for i in range(len(y)):
        if X[i][attribute_index] == value:
            new_x.append(X[i])
            new_y.append(y[i])

    return np.array(new_x), np.array(new_y)


def create_attribute_node(partial_X, partial_y, chosen_attribute_index, attribute_value, curr_node_index,
                          prev_node_index):
    """ Create an AttributeNode object which contain the partial_X and partial_y given.

    Parameters
    ----------
    partial_X: {array-like, sparse matrix}, shape (n_samples, n_features)
        Contains a partial samples from the train set.

    partial_y: (An array_like object of length n_samples)
        Contains the true class labels for all the samples in partial_X.

    chosen_attribute_index: int
        The chosen attribute index upon the node will be splited by.

    attribute_value: int
        The data value of the chosen attribute in this node.

    curr_node_index: int
        The index of the new AttributeNode which will be created

    prev_node_index: int
        The index of the previous node.

    Returns
    -------
        A new AttributeNode object initial with the given parameters.
    """

    if chosen_attribute_index < 0 or attribute_value <0 or curr_node_index < 0 or prev_node_index < 0:
        raise ValueError("All parameters should be positives")

    # Drop records where their value in chosen_attribute isn't equal to attribute_value
    x_y_tuple = drop_records(X=partial_X,
                             y=partial_y,
                             attribute_index=chosen_attribute_index,
                             value=attribute_value)
    # Create a new AttributeNode only is it has samples
    attributes_node = None
    if len(x_y_tuple[0]):
        attributes_node = AttributeNode(index=curr_node_index,
                                        attribute_value=attribute_value,
                                        prev_node=prev_node_index,
                                        layer=chosen_attribute_index,
                                        partial_x=x_y_tuple[0],
                                        partial_y=x_y_tuple[1])
    return attributes_node


def get_columns_type(X):
    """ Finding the type of each column in X

    Parameters
    ----------
    x: {array-like, sparse matrix}, shape (n_samples, n_features)
        The samples in the train set.

    Returns
    -------
        An array_like object of length n contains in each position the type of the corresponding attribute in X.

    """

    columns_type = []
    for dt in X.columns:
        if len(np.unique(X[dt])) / len(X) < 0.03:
            columns_type.append("category")
        else:
            columns_type.append(str(X[dt].dtype))
    return columns_type


def convert_numeric_values(chosen_split_points, chosen_attribute, partial_X):
    """ After finding the chosen split points for the given attribute, this function does the actual discretization.
        For each data value in the chosen attribute, this function covert it to number between 0 and
        len(chosen_split_points) by using the _find_split_position function.

    Parameters
    ----------
    partial_X: {array-like, sparse matrix}, shape (n_samples, n_features)
        Contains a partial samples from the train set.

    chosen_split_points: (An array_like object of length n)
        Contains all the chosen split points founded significant for the chosen_attribute.

    chosen_attribute: int
        The index of the attribute upon the network will be splited by.


    """

    chosen_split_points.sort()

    # Convert each value in record[chosen_attribute] to a number between 0 and len(chosen_split_points)
    for record in partial_X:
        record[chosen_attribute] = find_split_position(value=record[chosen_attribute],
                                                       positions=chosen_split_points)


def calculate_second_best_attribute_of_last_layer(attributes_mi: dict):
    """ This function finds and return the attribute index of the second best conditional mutual information
        based on the given dictionary.

    Parameters
    ----------
    attributes_mi: dict
        dictionary represent the conditional mutual information of each attribute.

    Returns
    -------
        The attribute index of the second best conditional mutual information
    """
    if attributes_mi is None or not bool(attributes_mi):
        return -1, 0

    attributes_mi_copy = attributes_mi.copy()

    index_of_max_cmi = max(attributes_mi_copy, key=attributes_mi.get)
    attributes_mi_copy.pop(index_of_max_cmi)

    if not bool(attributes_mi_copy):
        return -1, 0

    index_of_second_best = max(attributes_mi_copy, key=attributes_mi.get)
    sec_best_att_cmi = attributes_mi_copy[index_of_second_best]

    if attributes_mi_copy[index_of_second_best] == 0:
        index_of_second_best = -1

    return index_of_second_best, sec_best_att_cmi
