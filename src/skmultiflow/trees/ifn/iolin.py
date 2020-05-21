import math
import pandas as pd
import numpy as np
from scipy import stats
import copy
import pickle
from abc import ABC, abstractmethod
from skmultiflow.data import SEAGenerator
from sklearn.utils.validation import check_X_y
from skmultiflow.trees import IfnClassifier
from skmultiflow.trees.ifn.meta_learning import MetaLearning
import skmultiflow.trees.ifn.utils as Utils
from skmultiflow.trees.ifn._ifn_network import HiddenLayer


class IncrementalOnlineNetwork(ABC):

    def __init__(self,
                 classifier,
                 path,
                 number_of_classes=2,
                 n_min=378,
                 n_max=math.inf,
                 alpha=0.99,
                 Pe=0.5,
                 init_add_count=10,
                 inc_add_count=50,
                 max_add_count=100,
                 red_add_count=75,
                 min_add_count=1,
                 max_window=1000,
                 data_stream_generator=SEAGenerator()):

        """
        Parameters
        ----------
        classifier :
        path : String
            A path to save the model.
        number_of_classes : int
            The number of classes in the target.
        n_min : int
            The number of the first example to be classified by the system.
        n_max : int
            The number of the last example to be classified by the system.
            (if unspecified, the system will run indefinitely).
        alpha : float
            Significance level
        Pe : float
            Maximum allowable prediction error of the model.
        init_add_count : int
            The number of new examples to be classified by the first model.
        inc_add_count : int
            Amount (percentage) to increase the number of examples between model re-constructions.
        max_add_count : int
            Maximum number of examples between model re-constructions.
        red_add_count : int
            Amount (percentage) to reduce the number of examples between model reconstructions.
        min_add_count : int
            Minimum number of examples between model re-constructions.
        max_window : int
            Maximum number of examples in a training window.
        data_stream_generator : stream generator
            Stream generator for the stream data
        """

        self.classifier = classifier
        self.path = path
        self.number_of_classes = number_of_classes
        self.n_min = n_min
        self.n_max = n_max
        self.alpha = alpha
        self.Pe = Pe
        self.init_add_count = init_add_count
        self.inc_add_count = inc_add_count
        self.max_add_count = max_add_count
        self.red_add_count = red_add_count
        self.min_add_count = min_add_count
        self.max_window = max_window
        self.window = None
        self.meta_learning = MetaLearning(alpha, number_of_classes)
        self.data_stream_generator = data_stream_generator
        self.data_stream_generator.prepare_for_use()
        self.counter = 1

    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, value):
        self._classifier = value

    @abstractmethod
    def generate(self):
        pass

    def _update_current_network(self, training_window_X, training_window_y):
        """ This method, according to "https://www.sciencedirect.com/science/article/abs/pii/S156849460800046X"
            activates another method (_check_split_validation) for checking the split validity of the current network.
            Afterwards, it replaces the last layer of the network if needed. Finally, it activates the
            (_new_split_process) procedure attempting to split the last layer (whether it was replaced or not)
            and add a new hidden layer to the network, if necessary.

        Parameters
        ----------
        training_window_X: {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples of the new window.
        training_window_y: array-like, shape = [n_samples]
            The target values of the new samples in the new window.
        """

        copy_network = self._check_split_validation(training_window_X=pd.DataFrame(training_window_X),
                                                    training_window_y=training_window_y)

        should_replace, significant_nodes_indexes = self._check_replacement_of_last_layer(copy_network=copy_network)

        if should_replace:
            self._replace_last_layer(significant_nodes_indexes=significant_nodes_indexes)
            self._new_split_process(training_window_X=training_window_X)

    def _check_split_validation(self, training_window_X, training_window_y):
        """ This method, according to "https://www.sciencedirect.com/science/article/abs/pii/S156849460800046X"
            is responsible to verify that the current split of each node actually contributes to the conditional mutual
            information calculated from the current training set.

        Parameters
        ----------
        training_window_X: {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples of the new window.
        training_window_y: array-like, shape = [n_samples]
            The target values of the new samples in the new window.

        Returns
        -------
            The copy network of the classifier field.

        """

        copy_network = IncrementalOnlineNetwork.clone_network(network=self.classifier.network,
                                                              training_window_X=training_window_X,
                                                              training_window_y=training_window_y)

        curr_layer = copy_network.root_node.first_layer
        curr_layer_in_original_network = self.classifier.network.root_node.first_layer
        un_significant_nodes = []
        un_significant_nodes_indexes = []

        while curr_layer is not None:
            for node in curr_layer.nodes:
                if node.is_terminal is False:
                    statistic, critical, cmi = \
                        self._calculate_conditional_mutual_information_based_on_the_new_window(node=node,
                                                                                               index=curr_layer.index)

                    if critical < statistic:  # significant
                        continue
                    else:
                        un_significant_nodes_indexes.append(node.index)
                        un_significant_nodes.append(node)

            self.classifier._set_terminal_nodes(nodes=un_significant_nodes,
                                                class_count=self.classifier.class_count)

            IncrementalOnlineNetwork.eliminate_nodes(nodes=set(un_significant_nodes_indexes),
                                                     layer=curr_layer_in_original_network.next_layer,
                                                     prev_layer=curr_layer_in_original_network)
            curr_layer = curr_layer.next_layer
            curr_layer_in_original_network = curr_layer_in_original_network.next_layer

        return copy_network

    def _check_replacement_of_last_layer(self, copy_network):
        """ This method, according to "https://www.sciencedirect.com/science/article/abs/pii/S156849460800046X"
            is applied in order to determine which attribute is most appropriate to correspond to the last (final)
            hidden layer. The attributes under consideration include the attribute that is already associated with
            the last layer or the second best attribute for the last layer of the previous model.
            The attribute selected for the last layer of the new model will be the one with the highest conditional
            mutual information based on the current training window.

        Parameters
        ----------
        copy_network: IfnNetwork
            a copy reference of the IfnNetwork of the classifier field based on the new training window.

        Returns
        -------
            Boolean indicating if the last layer of the network should be replace with a new split
            and the indexes of the significant nodes which should be splitted upon the new attribute.

        """

        should_replace = False
        conditional_mutual_information_first_best_att = self.classifier.last_layer_mi
        index_of_second_best_att = self.classifier.index_of_sec_best_att

        if index_of_second_best_att == -1:  # There's only one significant attribute
            return should_replace, None

        conditional_mutual_information_second_best_att, significant_nodes_indexes = \
            self._calculate_cmi_of_sec_best_attribute(copy_network=copy_network,
                                                      sec_best_index=index_of_second_best_att)

        if conditional_mutual_information_first_best_att < conditional_mutual_information_second_best_att:
            should_replace = True

        return should_replace, significant_nodes_indexes

    def _calculate_cmi_of_sec_best_attribute(self, copy_network, sec_best_index):
        """ This method calculate the conditional mutual information of the sec best attribute based on the
            new training window.

        Parameters
        ----------
        copy_network: IfnNetwork
            a copy reference of the IfnNetwork of the classifier field based on the new training window.

        sec_best_index: int
            The index of the second best attribute in the training set to split the last layer by.

        Returns
        -------
            The calculated conditional mutual information of the second best attribute
            and the indexes of the significant nodes which should be splitted upon the new attribute.
        """

        conditional_mutual_information = 0
        significant_nodes_indexes = []
        curr_layer = copy_network.root_node.first_layer

        if curr_layer.next_layer is not None:
            while curr_layer.next_layer.next_layer is not None:  # loop until last split
                curr_layer = curr_layer.next_layer
                if curr_layer.next_layer is None:
                    break

        last_layer_nodes = curr_layer.nodes

        for node in last_layer_nodes:
            statistic, critical, cmi = \
                self._calculate_conditional_mutual_information_based_on_the_new_window(node=node,
                                                                                       index=sec_best_index)
            if critical < statistic:  # significant
                conditional_mutual_information += conditional_mutual_information + cmi
                significant_nodes_indexes.append(node.index)

        return conditional_mutual_information, set(significant_nodes_indexes)

    def _calculate_conditional_mutual_information_based_on_the_new_window(self, node, index):
        """ This function calculate the conditional mutual information of each node based on the
            current window after the network has been cloned using _clone_network method.

        Parameters
        ----------
        node: AttributeNode
            An AttributeNode represent a node in network.
        index: int
            Of the given node.

        Returns
        -------
            The statistic bases on the partial_X and partial_y in the node.
            The critical and the calculated mutual information.
        """

        X = node.partial_x
        y = node.partial_y
        attribute_data = list(X[:, index])
        unique_values = np.unique(attribute_data)
        conditional_mutual_information = \
            self.classifier._calculate_conditional_mutual_information(X=attribute_data,
                                                                      y=y)

        statistic = 2 * np.log(2) * len(y) * conditional_mutual_information
        critical = stats.chi2.ppf(self.alpha, ((self.number_of_classes - 1) * (len(unique_values) - 1)))

        return statistic, critical, conditional_mutual_information

    def _replace_last_layer(self, significant_nodes_indexes):
        """ This function replace the split of the last layer with the second based attribute of the previous model.

        Parameters
        ----------
        significant_nodes_indexes: set
            The indexes of the significant node in the before last layer which can be splitted by the second based
            attribute.

        """

        curr_layer = self.classifier.network.root_node.first_layer
        is_continuous = self.classifier.sec_att_split_points is not None
        index_of_sec_best_att = self.classifier.index_of_sec_best_att

        while curr_layer.next_layer.next_layer is not None:  # loop until last split
            curr_layer = curr_layer.next_layer

        new_layer_nodes = []
        terminal_nodes = []
        last_layer_nodes = curr_layer.nodes
        curr_node_index = max([node.index for node in curr_layer.nodes]) + 1

        for node in last_layer_nodes:
            if node.index in significant_nodes_indexes:
                if is_continuous:
                    Utils.convert_numeric_values(chosen_split_points=self.classifier.sec_att_split_points,
                                                 chosen_attribute=index_of_sec_best_att,
                                                 partial_X=node.partial_X)

                unique_values = np.unique(list(node.partial_X[:, index_of_sec_best_att]))

                for i in unique_values:  # create nodes for each unique value
                    attribute_node = Utils.create_attribute_node(partial_X=node.partial_X,
                                                                 partial_y=node.partial_y,
                                                                 chosen_attribute_index=index_of_sec_best_att,
                                                                 attribute_value=i,
                                                                 curr_node_index=curr_node_index,
                                                                 prev_node_index=node.index)
                    new_layer_nodes.append(attribute_node)
                    curr_node_index += curr_node_index + 1

            terminal_nodes.append(node)

        # create and link the new last layer to the network
        new_last_layer = HiddenLayer(index_of_sec_best_att)
        new_last_layer.is_continuous = is_continuous

        if new_last_layer.is_continuous is True:
            new_last_layer.split_points = self.classifier.sec_att_split_points

        new_last_layer.nodes = new_layer_nodes
        curr_layer.next_layer = new_last_layer

        # set all the nodes to be terminals
        self.classifier._set_terminal_nodes(nodes=terminal_nodes,
                                            class_count=self.classifier.class_count)

    def _new_split_process(self, training_window_X):

        training_window_X_df = pd.DataFrame(training_window_X)
        curr_layer = self.classifier.network.root_node.first_layer
        last_layer = None
        chosen_attributes = []

        while curr_layer is not None:  # collect of the chosen attributes of the splitting points in the network.
            chosen_attributes.append(curr_layer.index)
            if curr_layer.next_layer is None:
                last_layer = curr_layer
            curr_layer = curr_layer.next_layer

        chosen_attributes = set(chosen_attributes)
        attributes_indexes = list(range(0, len(training_window_X[0])))

        columns_type = Utils.get_columns_type(training_window_X_df)

        remaining_attributes = set(attributes_indexes) - set(chosen_attributes)
        remaining_attributes = list(remaining_attributes)

        global_chosen_attribute, attributes_mi, significant_attributes_per_node = \
            self.classifier.choose_split_attribute(attributes_indexes=remaining_attributes,
                                                   columns_type=columns_type,
                                                   nodes=last_layer.get_nodes())

        curr_node_index = max([node.index for node in last_layer.nodes]) + 1
        nodes_list = []
        terminal_nodes = []
        if global_chosen_attribute != -1:
            is_continuous = 'category' not in columns_type[global_chosen_attribute]
            node_index = 0
            for node in last_layer.get_nodes():
                if is_continuous:
                    if node in set(self.classifier.nodes_splitted_per_attribute[global_chosen_attribute]):
                        attributes_mi_per_node = 1
                    else:
                        attributes_mi_per_node = 0
                else:
                    attributes_mi_per_node = significant_attributes_per_node[global_chosen_attribute][node_index]
                node_index += 1

                if attributes_mi_per_node > 0:
                    if is_continuous:
                        Utils.convert_numeric_values(
                            chosen_split_points=self.classifier.split_points[global_chosen_attribute],
                            chosen_attribute=global_chosen_attribute,
                            partial_X=node.partial_X)
                    partial_X = node.partial_x
                    partial_y = node.partial_y
                    attribute_data_in_node = list(partial_X[:, global_chosen_attribute])
                    unique_values = np.unique(attribute_data_in_node)
                    prev_node = node.index
                    for i in unique_values:
                        attribute_node = Utils.create_attribute_node(partial_X=partial_X,
                                                                     partial_y=partial_y,
                                                                     chosen_attribute_index=global_chosen_attribute,
                                                                     attribute_value=i,
                                                                     curr_node_index=curr_node_index,
                                                                     prev_node_index=prev_node)
                        nodes_list.append(attribute_node)
                        terminal_nodes.append(attribute_node)
                        curr_node_index += 1
                else:
                    terminal_nodes.append(node)

            new_layer = HiddenLayer(global_chosen_attribute)
            last_layer.next_layer = new_layer
            new_layer.nodes = nodes_list
            if is_continuous:
                new_layer.is_continuous = True
                new_layer.split_points = self.classifier.split_points[global_chosen_attribute]

            self.classifier._set_terminal_nodes(nodes=terminal_nodes,
                                                class_count=self.classifier.class_count)

    def _induce_new_model(self, training_window_X, training_window_y):
        """ This method create a new network by calling fit method of IfnClassifier based on the training_window_X and
            training_window_y samples.

        Parameters
        ----------
        training_window_X: {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples of the new window.
        training_window_y: array-like, shape = [n_samples]
            The target values of the new samples in the new window.
        """
        # training_window_X_df = pd.DataFrame(training_window_X)
        self.classifier = self.classifier.partial_fit(training_window_X, training_window_y)
        path = self.path + "/" + str(self.counter) + ".pickle"
        pickle.dump(self.classifier, open(path, "wb"))
        self.counter = self.counter + 1

    @staticmethod
    def eliminate_nodes(nodes, layer, prev_layer):
        """ This function eliminate all the nodes in the given nodes param from the given layer.

        Parameters
        ----------
        nodes: set
            List of the node which need to be remove from the layer.
        layer: HiddenLayer
            The HiddenLayer which contains the nodes.
        prev_layer: HiddenLayer
            The previous layer of layer.

        """
        next_layer_nodes = []
        nodes_to_eliminate = []

        if nodes is None or len(nodes) == 0 or layer is None or prev_layer is None:
            return

        curr_layer_nodes = layer.nodes
        for node in curr_layer_nodes:
            if node.prev_node in nodes:
                nodes_to_eliminate.append(node)
                next_layer_nodes.append(node)

        for node in nodes_to_eliminate:
            layer.nodes.remove(node)

        if len(layer.nodes) == 0:
            prev_layer.next_layer = layer.next_layer

        IncrementalOnlineNetwork.eliminate_nodes(nodes=set(next_layer_nodes),
                                                 layer=layer.next_layer,
                                                 prev_layer=layer)

    @staticmethod
    def clone_network(network, training_window_X, training_window_y):
        """ This method copy the IfnNetwork from the classifier field and replace the training samples inside the
            nodes with the given training samples as parameters.

        Parameters
        ----------
        training_window_X: {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples of the new window.
        training_window_y: array-like, shape = [n_samples]
            The target values of the new samples in the new window.

        Returns
        -------
            The copy network of the classifier field.

        """

        copy_network = copy.copy(network)
        training_window_X_copy = training_window_X.copy()
        training_window_y_copy = training_window_y.copy()

        training_window_X_copy, training_window_y_copy = \
            check_X_y(training_window_X_copy, training_window_y_copy, accept_sparse=True)

        curr_layer = copy_network.root_node.first_layer
        is_first_layer = True
        converted = False
        nodes_data = {}
        while curr_layer is not None:
            for node in curr_layer.nodes:
                if is_first_layer:
                    if curr_layer.is_continuous and not converted:
                        Utils.convert_numeric_values(chosen_split_points=curr_layer.split_points,
                                                     chosen_attribute=curr_layer.index,
                                                     partial_X=training_window_X_copy)
                        converted = True

                    partial_X, partial_y = Utils.drop_records(X=training_window_X_copy,
                                                              y=training_window_y_copy,
                                                              attribute_index=curr_layer.index,
                                                              value=node.attribute_value)
                    node.partial_x = partial_X
                    node.partial_y = partial_y
                    nodes_data[node.index] = [partial_X, partial_y]

                else:
                    X = nodes_data[node.prev_node][0]
                    y = nodes_data[node.prev_node][1]
                    if curr_layer.is_continuous and not converted:
                        Utils.convert_numeric_values(chosen_split_points=curr_layer.split_points,
                                                     chosen_attribute=curr_layer.index,
                                                     partial_X=X)
                        converted = True

                    partial_X, partial_y = Utils.drop_records(X=X,
                                                              y=y,
                                                              attribute_index=curr_layer.index,
                                                              value=node.attribute_value)
                    node.partial_x = partial_X
                    node.partial_y = partial_y
                    nodes_data[node.index] = [partial_X, partial_y]

            is_first_layer = False
            converted = False
            curr_layer = curr_layer.next_layer

        return copy_network
