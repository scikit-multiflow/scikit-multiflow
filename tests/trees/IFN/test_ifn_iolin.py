import copy

import numpy as np
import pytest, os

from sklearn.utils import check_X_y

from skmultiflow.trees.IFN.data_processing import DataProcessor
from skmultiflow.trees.IFN.IFN_network import IfnNetwork, IfnHiddenLayer, IfnAttributeNode
from skmultiflow.trees import IfnBasicIncremental

dataset_path = "src/skmultiflow/data/datasets/elec.csv"


def _setup_eliminate_nodes_test_env():
    network = IfnNetwork()
    network.build_target_layer([0, 1])

    first_layer = IfnHiddenLayer(1)

    node1 = IfnAttributeNode(index=1,
                             attribute_value=0,
                             prev_node=0,
                             layer=1,
                             partial_x=None,
                             partial_y=None,
                             is_terminal=False)
    node2 = IfnAttributeNode(index=2,
                             attribute_value=1,
                             prev_node=0,
                             layer=1,
                             partial_x=None,
                             partial_y=None,
                             is_terminal=False)

    first_layer.nodes = [node1, node2]

    network.root_node.first_layer = first_layer

    second_layer = IfnHiddenLayer(2)

    node1 = IfnAttributeNode(index=3,
                             attribute_value=0,
                             prev_node=1,
                             layer=2,
                             partial_x=None,
                             partial_y=None,
                             is_terminal=False)
    node2 = IfnAttributeNode(index=4,
                             attribute_value=1,
                             prev_node=1,
                             layer=2,
                             partial_x=None,
                             partial_y=None,
                             is_terminal=True)
    node3 = IfnAttributeNode(index=5,
                             attribute_value=0,
                             prev_node=2,
                             layer=2,
                             partial_x=None,
                             partial_y=None,
                             is_terminal=False)

    node4 = IfnAttributeNode(index=6,
                             attribute_value=1,
                             prev_node=2,
                             layer=2,
                             partial_x=None,
                             partial_y=None,
                             is_terminal=True)

    second_layer.nodes = [node1, node2, node3, node4]
    first_layer.next_layer = second_layer

    third_layer = IfnHiddenLayer(3)

    node1 = IfnAttributeNode(index=7,
                             attribute_value=0,
                             prev_node=3,
                             layer=3,
                             partial_x=None,
                             partial_y=None,
                             is_terminal=True)
    node2 = IfnAttributeNode(index=8,
                             attribute_value=1,
                             prev_node=3,
                             layer=3,
                             partial_x=None,
                             partial_y=None,
                             is_terminal=True)
    node3 = IfnAttributeNode(index=9,
                             attribute_value=0,
                             prev_node=5,
                             layer=3,
                             partial_x=None,
                             partial_y=None,
                             is_terminal=True)

    node4 = IfnAttributeNode(index=10,
                             attribute_value=1,
                             prev_node=5,
                             layer=3,
                             partial_x=None,
                             partial_y=None,
                             is_terminal=True)

    third_layer.nodes = [node1, node2, node3, node4]
    second_layer.next_layer = third_layer
    return network


def test_eliminate_all_nodes_in_layer():
    network= _setup_eliminate_nodes_test_env()

    IfnBasicIncremental.eliminate_nodes(nodes={3, 5},
                                        layer=network.root_node.first_layer.next_layer.next_layer,
                                        prev_layer=network.root_node.first_layer.next_layer)

    new_third_layer = network.root_node.first_layer.next_layer.next_layer

    assert new_third_layer is None


def test_eliminate_some_nodes_in_layer():
    network = _setup_eliminate_nodes_test_env()
    third_layer_nodes = network.root_node.first_layer.next_layer.next_layer.nodes
    nodes_to_remains = []

    for node in third_layer_nodes:
        if node.index == 9 or node.index == 10:
            nodes_to_remains.append(node)

    IfnBasicIncremental.eliminate_nodes(nodes={3},
                                        layer=network.root_node.first_layer.next_layer.next_layer,
                                        prev_layer=network.root_node.first_layer.next_layer)

    new_third_layer = network.root_node.first_layer.next_layer.next_layer
    third_layer_nodes = network.root_node.first_layer.next_layer.next_layer.nodes

    assert new_third_layer is not None
    assert len(third_layer_nodes) == 2
    assert np.array_equal(nodes_to_remains, third_layer_nodes)


def test_eliminate_nodes_with_invalid_input():
    network = _setup_eliminate_nodes_test_env()
    prev_number_of_nodes = len(network.root_node.first_layer.next_layer.next_layer.nodes)
    IfnBasicIncremental.eliminate_nodes(nodes=set(),
                                        layer=network.root_node.first_layer.next_layer.next_layer,
                                        prev_layer=network.root_node.first_layer.next_layer)

    assert prev_number_of_nodes == len(network.root_node.first_layer.next_layer.next_layer.nodes)

    IfnBasicIncremental.eliminate_nodes(nodes={3},
                                        layer=network.root_node.first_layer.next_layer.next_layer,
                                        prev_layer=None)

    assert prev_number_of_nodes == len(network.root_node.first_layer.next_layer.next_layer.nodes)


def test_clone_network():
    network = _setup_eliminate_nodes_test_env()
    dp = DataProcessor()
    x_train, x_test, y_train, y_test = \
        dp.convert(
            csv_file_path=dataset_path,
            test_size=0.3)

    x_train, y_train = check_X_y(x_train, y_train, accept_sparse=True)

    copy_network = IfnBasicIncremental.clone_network(network=network,
                                                     training_window_X=x_train,
                                                     training_window_y=y_train)

    assert copy_network.root_node.first_layer.index == 1
    assert len(copy_network.root_node.first_layer.next_layer.nodes) == 4


def test_clone_network_invalid_input():
    network = IfnNetwork()
    with pytest.raises(AttributeError):
        IfnBasicIncremental.clone_network(network=network,
                                          training_window_X=None,
                                          training_window_y=[0, 1])
        IfnBasicIncremental.clone_network(network=None,
                                          training_window_X=None,
                                          training_window_y=[0, 1])
