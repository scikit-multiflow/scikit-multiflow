"""
Author: Mor Avitan, Itay Carmi and Omri Naor
Original code and method by: Prof' Mark Last
License: BSD 3 clause
"""
import numpy as np

import pandas as pd

from scipy import stats

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from skmultiflow.trees.IFN.IFN_network import IfnNetwork, IfnHiddenLayer
from skmultiflow.trees.IFN import IFN_utils as utils
from skmultiflow.core import BaseSKMObject, ClassifierMixin

import math
import collections
import time


class IfnClassifier(BaseSKMObject, ClassifierMixin):
    """ A template estimator to be used as a reference implementation.

    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    columns_type: list
        List of the columns type in the X data the classifier will receive.
        the list should contain strings only. 'category' or 'int64'
    alpha: float, default='0.99'
        A parameter used for the significance level of the likelihood-ratio tests.

    max_number_of_layers: int, default=math.inf
        The maximum number of layers the network will have.

    window_size: int, default=100
        The size of the stream window to be procced.
    """

    def __init__(self, columns_type, alpha=0.99, max_number_of_layers=math.inf, window_size=100):
        if 0 <= alpha < 1:
            self.alpha = alpha
            self.max_number_of_layers = max_number_of_layers
            self.window_size = window_size
            self.X_batch = []
            self.y_batch = []
            self.i = 0
            self.is_fitted = False
            self.columns_type = columns_type
            self.training_error = 0
            self.cmi_sec_best_att = 0
            self.index_of_sec_best_att = 0
            self.sec_att_split_points = None
            self.class_count = None
            self.total_records = 0
            # Number of classes in the target
            self.num_of_classes = 0
            # Dictionary that contains the unique value for each attribute
            self.unique_values_per_attribute = {}
            # Dictionary that contains all the split points for each attribute
            self.split_points = {}
            # Dictionary that contains for each node and attribute all the split points founded significant
            self.nodes_splitted_per_attribute = {}
            # Dictionary that contains for each numeric attribute it's data interval
            self.intervals_per_attribute = {}
            # array-like that contains for each group of split points the nodes founded significant and the
            # conditional mutual information.
            # Example: [[frozenset: {'1,7,10'}, list<AttributeNode>, float: conditional mutual information]..]
            self.splitted_nodes_by_split_points = []
        else:
            raise ValueError("Enter a valid alpha between 0 to 1")
        self.network = IfnNetwork()

    def fit(self, X, y, classes=None, sample_weight=None):

        """A reference implementation of a fitting function.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        classes: numpy.array
            Contains the class values in the stream. If defined, will be used
            to define the length of the arrays returned by `predict_proba`
        sample_weight: float or array-like
            Samples weight. If not provided, uniform weights are assumed.

        Returns
        -------
            self
        """
        self.is_fitted = False
        start = time.time()
        print('Building the network...')

        cols = list(X.columns.values)
        cols = [str(i) for i in cols]

        X, y = check_X_y(X, y, accept_sparse=True)
        X_copy = X.copy()
        y_copy = y.copy()
        self.total_records = len(y)
        unique, counts = np.unique(np.array(y), return_counts=True)
        self.class_count = np.asarray((unique, counts)).T
        self.num_of_classes = len(unique)

        # create list the holds the attributes indexes
        attributes_indexes = list(range(0, len(X[0])))

        layer = 'first'
        number_of_layers = 0
        curr_node_index = 1
        current_layer = None
        last_layer_mi = {}

        self.network.build_target_layer(unique)

        # Define for each numeric attribute it's interval for the discretization procedure
        self.intervals_per_attribute = self._define_interval_for_numeric_feature(X, y, attributes_indexes,
                                                                                 self.columns_type)

        significant_attributes_per_node = {}

        with open("output.txt", "w+") as f:
            f.write('Output data for dataset: \n\n')
            f.write('Total instances: ' + str(self.total_records) + '\n')
            f.write('Number of candidate input attributes is: ' + str(len(attributes_indexes)) + '\n')
            f.write('Minimum confidence level is: ' + str(self.alpha) + '\n\n')
            f.close()

        # Build the network while:
        # 1. The maximum number of hidden layers is not exceeded
        # 2. There are more attribute to split the network by them
        while len(attributes_indexes) > 0 and number_of_layers < self.max_number_of_layers:
            chosen_split_points = []
            if current_layer is not None:
                global_chosen_attribute, attributes_mi, significant_attributes_per_node = \
                    self._choose_split_attribute(attributes_indexes=attributes_indexes,
                                                 columns_type=self.columns_type,
                                                 nodes=current_layer.get_nodes())
            # first layer
            else:
                global_chosen_attribute, attributes_mi, not_relevant = \
                    self._choose_split_attribute(attributes_indexes=attributes_indexes,
                                                 columns_type=self.columns_type,
                                                 X=X,
                                                 y=y)

            # there isn't an attribute to split the network by
            if global_chosen_attribute == -1:
                if curr_node_index == 1:
                    print('No Nodes at the network. choose smaller alpha')
                    return
                    # sys.exit()
                utils.write_details_to_file(layer_position=layer,
                                            attributes_cmi=attributes_mi,
                                            chosen_attribute_index=global_chosen_attribute,
                                            chosen_attribute=cols[global_chosen_attribute])

                break

            nodes_list = []

            is_continuous = 'category' not in self.columns_type[global_chosen_attribute]
            # if chosen attribute is continuous we convert the partial x values by the splits values
            if is_continuous:
                chosen_split_points = self.split_points[global_chosen_attribute]
                self._convert_numeric_values(chosen_split_points=chosen_split_points,
                                             chosen_attribute=global_chosen_attribute,
                                             layer=current_layer,
                                             partial_X=X)

            # Create new hidden layer of the maximal mutual information attribute and set the layer nodes
            insignificant_nodes = []
            if current_layer is not None:
                node_index = 0
                for node in current_layer.get_nodes():
                    # Check if the node is significant by the chosen attribute
                    # For both cases: continuous feature and categorical feature
                    if is_continuous:
                        if node in set(self.nodes_splitted_per_attribute[global_chosen_attribute]):
                            attributes_mi_per_node = 1
                        else:
                            attributes_mi_per_node = 0
                    else:
                        attributes_mi_per_node = significant_attributes_per_node[global_chosen_attribute][node_index]
                    node_index += 1
                    # If global chosen attribute is significant at this node then we split the node by this attribute
                    if attributes_mi_per_node > 0:
                        partial_X = node.partial_x
                        partial_y = node.partial_y
                        attribute_data_in_node = list(partial_X[:, global_chosen_attribute])
                        unique_values = np.unique(attribute_data_in_node)
                        prev_node = node.index
                        for i in unique_values:
                            attribute_node = utils.create_attribute_node(partial_X=partial_X,
                                                                         partial_y=partial_y,
                                                                         chosen_attribute_index=global_chosen_attribute,
                                                                         attribute_value=i,
                                                                         curr_node_index=curr_node_index,
                                                                         prev_node_index=prev_node)
                            nodes_list.append(attribute_node)
                            curr_node_index += 1
                    # If the node isn't significant we will set it as terminal node later
                    else:
                        insignificant_nodes.append(node)
            # First layer
            else:
                prev_node = 0
                for i in self.unique_values_per_attribute[global_chosen_attribute]:
                    attribute_node = utils.create_attribute_node(partial_X=X,
                                                                 partial_y=y,
                                                                 chosen_attribute_index=global_chosen_attribute,
                                                                 attribute_value=i,
                                                                 curr_node_index=curr_node_index,
                                                                 prev_node_index=prev_node)
                    nodes_list.append(attribute_node)
                    curr_node_index += 1

            next_layer = IfnHiddenLayer(global_chosen_attribute)

            # If we're in the first layer
            if current_layer is None:
                self.network.root_node.first_layer = next_layer
            else:
                current_layer.next_layer = next_layer

            next_layer.set_nodes(nodes_list)

            if is_continuous:
                next_layer.is_continuous = True
                next_layer.split_points = chosen_split_points

            # Set the un significant node as terminal nodes
            insignificant_nodes_set = list(set(insignificant_nodes))
            self._set_terminal_nodes(insignificant_nodes_set, self.class_count)

            current_layer = next_layer
            number_of_layers += 1

            utils.write_details_to_file(layer_position=layer,
                                        attributes_cmi=attributes_mi,
                                        chosen_attribute_index=global_chosen_attribute,
                                        chosen_attribute=cols[global_chosen_attribute])

            # overrides the value until the last iteration
            self.last_layer_mi = attributes_mi[global_chosen_attribute]
            last_layer_mi = attributes_mi.copy()
            layer = 'next'

            attributes_indexes.remove(global_chosen_attribute)
            self.split_points.clear()
            self.nodes_splitted_per_attribute.clear()
            significant_attributes_per_node.clear()

        # Network building is done
        # Set the remaining nodes as terminals
        self._set_terminal_nodes(nodes=current_layer.get_nodes(),
                                 class_count=self.class_count)

        with open('output.txt', 'a') as f:
            f.write('Total nodes created:' + str(curr_node_index) + "\n")
            end = time.time()
            f.write("Running time: " + str(round(end - start, 3)) + " Sec")
            f.close()

        self.is_fitted = True
        print("Done. Network is fitted")

        self.training_error = self.calculate_error_rate(X=X_copy, y=y_copy)
        self.index_of_sec_best_att, self.cmi_sec_best_att = \
            utils.calculate_second_best_attribute_of_last_layer(attributes_mi=last_layer_mi)

        if self.index_of_sec_best_att != -1 and 'category' not in self.columns_type[self.index_of_sec_best_att]:
            self.sec_att_split_points = self.split_points[self.index_of_sec_best_att]

        self.split_points.clear()
        self.nodes_splitted_per_attribute.clear()
        significant_attributes_per_node.clear()

        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """ Partially (incrementally) fit the model.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.

        y: numpy.ndarray of shape (n_samples)
            An array-like with the labels of all samples in X.

        classes: Not used (default=None)

        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed.

        Returns
        -------
            self

        """
        N, D = X.shape
        isEnoughSamples = False

        if self.i == 0:
            # No models yet -- initialize
            self.X_batch = np.zeros((self.window_size, D))
            self.y_batch = np.zeros(self.window_size)
            self.sample_weight = np.zeros(N)
            self.i = 0

        for n in range(N):
            # For each instance ...
            self.X_batch[self.i] = X[n]
            self.y_batch[self.i] = y[n]
            # self.sample_weight[self.i] = sample_weight[n] if sample_weight else 1.0
            self.i = self.i + 1

            if self.i == self.window_size:
                isEnoughSamples = True
                # Train it
                X_batch_df = pd.DataFrame(self.X_batch)
                self.fit(X=X_batch_df, y=self.y_batch, classes=classes, sample_weight=sample_weight)
                # Reset the window
                self.i = 0

        # if not isEnoughSamples:
        # TODO maybe change to return 0
        # print("There are not enough samples to build a network")

        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted')
        predicted = []
        for record in X:
            curr_layer = self.network.root_node.first_layer
            prev_node_index = 0
            found_terminal_node = False
            while curr_layer is not None and not found_terminal_node:
                record_value = record[curr_layer.index]
                if curr_layer.is_continuous:
                    record_value = utils.find_split_position(value=record_value,
                                                             positions=curr_layer.split_points)

                possible_nodes = [node for node in curr_layer.nodes if node.prev_node == prev_node_index]
                iteration_number = 1
                for node in possible_nodes:
                    if node.attribute_value == record_value or \
                            (iteration_number == len(possible_nodes) and node.attribute_value + 1 == record_value) or \
                            (iteration_number == 1 and node.attribute_value - 1 == record_value):
                        chosen_node = node
                        if chosen_node.is_terminal:
                            max_weight = -math.inf
                            predicted_class = -math.inf
                            for class_index, weight_prob_pair in chosen_node.weight_probability_pair.items():
                                if weight_prob_pair[0] > max_weight:
                                    max_weight = weight_prob_pair[0]
                                    predicted_class = class_index
                            predicted.append(predicted_class)
                            found_terminal_node = True
                        else:
                            curr_layer = curr_layer.next_layer
                            prev_node_index = chosen_node.index
                        break

                    iteration_number += 1

        index = 1
        with open('predict.txt', 'w') as f:
            for row in predicted:
                f.write(str(index) + '. ' + str(row) + '\n')
                index += 1
            f.close()

        return np.array(predicted).astype(int)

    def predict_proba(self, X):
        """ A reference implementation of a predicting probabilities function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        predicted = []
        for record in X:
            curr_layer = self.network.root_node.first_layer
            prev_node_index = 0
            found_terminal_node = False
            while curr_layer is not None and not found_terminal_node:
                record_value = record[curr_layer.index]
                if curr_layer.is_continuous is not False:
                    record_value = utils.find_split_position(value=record_value,
                                                             positions=curr_layer.split_points)
                possible_nodes = [node for node in curr_layer.nodes if node.prev_node == prev_node_index]
                iteration_number = 1
                for node in possible_nodes:
                    if node.attribute_value == record_value or \
                            (iteration_number == len(possible_nodes) and node.attribute_value + 1 == record_value) or \
                            (iteration_number == 1 and node.attribute_value - 1 == record_value):
                        chosen_node = node
                        if chosen_node.is_terminal:
                            found_terminal_node = True
                            weights_of_node = []
                            for class_index, weight_prob_pair in chosen_node.weight_probability_pair.items():
                                weights_of_node.append((round(weight_prob_pair[1], 3)))
                            predicted.append(weights_of_node)
                        else:
                            curr_layer = curr_layer.next_layer
                            prev_node_index = chosen_node.index
                        break

                    iteration_number += 1

        index = 1
        with open('tmp/predict.txt', 'w') as f:
            for row in predicted:
                f.write(str(index) + '. ' + str(row) + '\n')
                index += 1
            f.close()

        return np.array(predicted).astype(int)

    def _choose_split_attribute(self, attributes_indexes, columns_type, nodes=None, X=None, y=None):
        """ Returns the most significant attribute upon all.
            The mose significant attribute is the one hold the higher conditional mutual information.

        Parameters
        ----------
        attributes_indexes: list
            Indexes of the remaining attributes the network hasn't being splitted by.

        columns_type: list
            Contains the type of each attribute.

        nodes: list, optional
            list of AttributeNode object in the current layer.
            None if the number of layer in the network is one.


        x: {array-like, sparse matrix}, shape (n_samples, n_feature), optional
            Contains the data of one feature overall samples in the train set.
            None if there is more than one layer in the network.

        y: (An array_like object of length n_samples)
            Contains the true class labels for all the samples in X.
            None if there is more than one layer in the network.

        Returns
        -------
            The chosen attribute index
            Dictionary mapping for each attribute the calculated conditional mutual information over all nodes/X
            Dictionary mapping for each node and attribute the calculated conditional mutual information.

        """
        attributes_mi = {}
        # for each node save the mutual information of each attribute index
        node_mi_per_attribute = {}
        # get the attribute that holds the maximal mutual information
        for attribute_index in attributes_indexes:
            node_mi_per_attribute[attribute_index] = []
            is_continuous = 'category' not in columns_type[attribute_index]
            # first layer
            if nodes is None:
                if is_continuous:
                    self._choose_split_numeric_attribute(attribute_index=attribute_index,
                                                         attributes_mi=attributes_mi)
                else:
                    self._choose_split_categorical_attribute(X=X,
                                                             y=y,
                                                             attribute_index=attribute_index,
                                                             attributes_mi=attributes_mi)
            else:
                if is_continuous:
                    splitted_nodes = self._choose_split_numeric_attribute(attribute_index=attribute_index,
                                                                          attributes_mi=attributes_mi,
                                                                          nodes=nodes)
                    self.nodes_splitted_per_attribute[attribute_index] = splitted_nodes
                else:
                    for node in nodes:
                        node_mi = self._choose_split_categorical_attribute(X=node.partial_x,
                                                                           y=node.partial_y,
                                                                           attribute_index=attribute_index,
                                                                           attributes_mi=attributes_mi)
                        node_mi_per_attribute[attribute_index].append(node_mi)

        chosen_attribute_index = max(attributes_mi, key=attributes_mi.get)

        if attributes_mi[chosen_attribute_index] <= 0:
            chosen_attribute_index = -1

        return chosen_attribute_index, attributes_mi, node_mi_per_attribute

    def _choose_split_categorical_attribute(self, X, y, attribute_index, attributes_mi):
        """ Calculate the mutual information of the given categorical attribute in the train set.

        Parameters
        ---------
        x: {array-like, sparse matrix}, shape (n_samples, n_feature)
            Contains the data of one feature overall samples in the train set.

        y: (An array_like object of length n_samples)
            Contains the true class labels for all the samples in X.

        attribute_index: int
            The index of the attribute being checked.

        attributes_mi: (dictionary) {attribute_index : conditional mutual information}
            Contains the conditional mutual information of each attribute.

        Returns
        -------
            The conditional mutual information of attribute being check in respect to X and y
        """

        node_mi = 0
        attribute_data = list(X[:, attribute_index])
        self.unique_values_per_attribute[attribute_index] = np.unique(attribute_data)

        mutual_info_score = self._calculate_conditional_mutual_information(attribute_data, y)
        statistic = 2 * np.log(2) * self.total_records * mutual_info_score
        critical = stats.chi2.ppf(self.alpha, ((self.num_of_classes - 1) *
                                               ((len(self.unique_values_per_attribute[attribute_index])) - 1)))

        if critical < statistic:
            # sum mutual information overall nodes
            if attribute_index in attributes_mi.keys():
                attributes_mi[attribute_index] += mutual_info_score
            else:
                attributes_mi[attribute_index] = mutual_info_score
            node_mi = mutual_info_score
        else:
            if attribute_index not in attributes_mi.keys():
                attributes_mi[attribute_index] = 0

        return node_mi

    def _choose_split_numeric_attribute(self, attribute_index, attributes_mi, nodes=None):
        """ Calculate the mutual information of the given numeric attribute in the train set.

        Parameters
        ----------
        attribute_index: int
            The index of the attribute being checked.

        attributes_mi: (dictionary) {attribute_index : conditional mutual information}
            Contains the conditional mutual information of each attribute.

        nodes: list, optional
            list of nodes in the current layer of the network


        Returns
        -------
            A list of split points of type float which founded significant in the current attribute.
            In case the attribute founded un-significant the list will be empty.
        """

        self.split_points[attribute_index] = []
        splitted_nodes = []
        # first layer
        if nodes is None:
            new_total_mi = self._discretization(attribute_index=attribute_index,
                                                interval=self.intervals_per_attribute[attribute_index], total_mi=[0])
            new_total_mi = new_total_mi[0]
        else:
            new_total_mi = self._discretization(attribute_index=attribute_index,
                                                interval=self.intervals_per_attribute[attribute_index], total_mi=[0],
                                                nodes=nodes)
            new_total_mi = new_total_mi[0]

            if bool(self.splitted_nodes_by_split_points):
                # total_mi = [el[2] for el in self.splitted_nodes_by_split_points]
                # max_mi = max(total_mi)
                # max_mi_index = total_mi.index(max_mi)
                # self.split_points[attribute_index] = list(self.splitted_nodes_by_split_points[max_mi_index][0])
                # new_total_mi = max_mi
                # splitted_nodes = list(self.splitted_nodes_by_split_points[max_mi_index][1])
                splitted_nodes = self.splitted_nodes_by_split_points.copy()
                self.splitted_nodes_by_split_points.clear()

        if bool(self.split_points[attribute_index]):  # there are split points
            attributes_mi[attribute_index] = new_total_mi
        else:
            attributes_mi[attribute_index] = 0

        return splitted_nodes

    def _discretization(self, attribute_index, interval, total_mi=None, nodes=None, prev_split_points=None):
        """ A recursive implementation of a discretization of the IFN algorithm according to the algorithm
            published in "Maimon, Oded, and Mark Last. "Knowledge discovery and data mining." Klewer Pub. Co (2001)."


        Parameters
        ----------
        attribute_index: int
            Index of the attribute being checked

        interval: {array-like, sparse matrix}, shape (n_samples, n_classes)
            Contains the data of one feature overall samples in the train set.

        total_mi: list of one element (default [0])
            The total conditional mutual information of the attribute being checked
            This variables is increasing at each recursive loop.

        splitted_nodes: list, optional
            list of nodes in the current layer of the network

        prev_split_points: frozenset, optional
            Set of the previous split points founded in the recursion

        Returns
        -------
            The new total mutual information of the attribute being checked
            :param nodes:

        """
        if total_mi is None:
            total_mi = [0]
        interval_values = [i[0] for i in interval]
        distinct_attribute_data = np.unique(interval_values)

        min_value = min(distinct_attribute_data)
        max_value = max(distinct_attribute_data)

        # mapping the mutual information to every possible split point
        split_point_mi_map = {}
        # mapping for each node the mutual information of every possible split point
        node_mi_per_threshold = {}
        # save the olf total mutual information in case no split point will be founded
        new_total_mi = total_mi
        # Counter for the number of nodes we don't need to check anymore

        iterator = iter(distinct_attribute_data)
        next(iterator)

        if prev_split_points is not None:
            curr_previous_split_points = list(prev_split_points)
        else:
            curr_previous_split_points = prev_split_points

        for T in iterator:
            if T in self.split_points[attribute_index]: continue
            if nodes is None:
                t_attribute_date, new_y = utils.split_data_to_two_intervals(interval=interval,
                                                                            T=T,
                                                                            min_value=min_value,
                                                                            max_value=max_value)
                if len(np.unique(t_attribute_date)) != 2:
                    break

                statistic, critical, t_mi = self._calculate_statistic_and_critical_for_interval(X=t_attribute_date,
                                                                                                y=new_y)
                # T in attribute is a possible split point
                if critical < statistic:
                    # For each point save it's mutual information
                    split_point_mi_map[T] = t_mi
            else:
                for node in nodes:
                    partial_X = node.partial_x
                    partial_y = node.partial_y
                    attribute_data = list(partial_X[:, attribute_index])
                    data_class_array = list(zip(attribute_data, partial_y))

                    t_attribute_date, new_y = utils.split_data_to_two_intervals(interval=data_class_array,
                                                                                T=T,
                                                                                min_value=min_value,
                                                                                max_value=max_value)

                    if len(np.unique(t_attribute_date)) != 2:
                        continue

                    statistic, critical, t_mi = self._calculate_statistic_and_critical_for_interval(X=t_attribute_date,
                                                                                                    y=new_y)

                    # T in attribute is a possible split point
                    if critical < statistic:
                        # for each point save it's mutual information
                        if node.index not in node_mi_per_threshold.keys():
                            node_mi_per_threshold[node.index] = {}
                        node_mi_per_threshold[node.index][T] = t_mi
                        if T not in split_point_mi_map.keys():
                            split_point_mi_map[T] = t_mi
                        else:
                            split_point_mi_map[T] += t_mi
                    else:
                        if node.index not in node_mi_per_threshold.keys():
                            node_mi_per_threshold[node.index] = {}
                        node_mi_per_threshold[node.index][T] = 0

        if bool(split_point_mi_map):  # if not empty
            # Find the split point which maximize the mutual information
            split_point = max(split_point_mi_map, key=split_point_mi_map.get)
            if split_point not in self.split_points[attribute_index]:
                self.split_points[attribute_index].append(split_point)
            if nodes is not None:
                if curr_previous_split_points is not None:
                    curr_previous_split_points.append(split_point)
                else:
                    curr_previous_split_points = [split_point]

            # Find the split point index in the interval using binary search
            l = [e[0] for e in interval]
            split_point_index = utils.binary_search(l, 0, len(l), split_point)
            # Split the interval into two intervals
            # smaller - includes all the elements where their value is smaller than split point
            interval_smaller = interval[0: split_point_index]
            # larger - includes all the elements where their value is equal or higher than split point
            interval_larger = interval[split_point_index:]

            # Found the nodes which are significant to the founded split point
            if nodes is not None:
                for node in nodes:
                    if node.index in node_mi_per_threshold.keys() \
                            and split_point in node_mi_per_threshold[node.index].keys() \
                            and node_mi_per_threshold[node.index][split_point] > 0:
                        self.splitted_nodes_by_split_points.append(node)
                self.splitted_nodes_by_split_points = list(set(self.splitted_nodes_by_split_points))

            new_total_mi[0] += split_point_mi_map[split_point]

            if bool(interval_smaller):
                self._discretization(attribute_index=attribute_index, interval=interval_smaller, total_mi=new_total_mi,
                                     nodes=nodes, prev_split_points=curr_previous_split_points)
            if bool(interval_larger):
                self._discretization(attribute_index=attribute_index, interval=interval_larger, total_mi=new_total_mi,
                                     nodes=nodes, prev_split_points=curr_previous_split_points)
        return new_total_mi

    def _calculate_statistic_and_critical_for_interval(self, X, y):
        """ calculate the statistic and critical for the data in interval X

        Parameters
        ---------
        x: {array-like, sparse matrix}, shape (n_samples, 1_feature)
            Contains the data of one feature overall samples in the train set.

        y: (An array_like object of length n_samples)
            Contains the true class labels for all the samples in X.

        Returns
        -------
            A float representing the statistic of the given interval X.
            A float representing the critical of the given interval y

        """

        if self.num_of_classes == 2:
            critical = stats.chi2.ppf(self.alpha, (self.num_of_classes - 1))
        else:
            rel_num_of_classes = len(np.unique(np.array(y)))
            critical = stats.chi2.ppf(self.alpha, (rel_num_of_classes - 1))

        t_mi = self._calculate_conditional_mutual_information(X=X, y=y)
        statistic = 2 * np.log(2) * self.total_records * t_mi

        return statistic, critical, t_mi

    def _calculate_conditional_mutual_information(self, X, y):
        """ Calculate the conditional mutual information of the feature given in x.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape (n_samples, 1_feature)
            Contains the data of one feature overall samples in the train set.

        y: (An array_like object of length n_samples)
            Contains the true class labels for all the samples in X.

        total_records: int
            Number of total records in the train set.

        Returns
        -------
            The total conditional mutual information of the feature.

        """
        partial_records = len(y)
        # count the number of classes (0 and 1)
        unique, counts = np.unique(np.array(y), return_counts=True)
        # <class, number_of_appearances>
        class_count = dict(zip(unique, counts))
        # count the number of distinct values in x
        unique, counts = np.unique(np.array(X), return_counts=True)
        # <value, number_of_appearances>
        data_count = dict(zip(unique, counts))

        data_dic = collections.defaultdict(int)

        # Count the number of appearances for each tuple x[i],y[i]
        for i in range(len(y)):
            data_dic[X[i], y[i]] = data_dic[X[i], y[i]] + 1

        total_cmi = 0

        # key = [feature_value,class], value = number of appearances of feature value and class together in x
        for key, value in data_dic.items():
            # Get the total number of class appearances
            curr_class_count = class_count[key[1]]
            # Get the total number of feature value appearances
            curr_data_count = data_count[key[0]]

            joint = value / self.total_records
            cond = value / partial_records
            cond_x = curr_data_count / partial_records
            cond_y = curr_class_count / partial_records

            mutual_information = joint * math.log(cond / (cond_x * cond_y), 2)
            total_cmi += mutual_information

        return total_cmi

    def _calculate_weights(self, y, class_count):
        """ Calculate the weights for each node in the last layer and the terminal nodes.

        Parameters
        ----------
        y: (An array_like object of length n_samples)
            Contains the true class labels for all the samples in X.

        class_count: (list of length n contain tuples)
            Contain list of tuples - (class value, number of appearances in the train set).


        Returns
        -------
            The weights for each class.

        """
        weights_per_class = {}
        for class_info in class_count:
            cut_len = len(np.extract(y == [class_info[0]], y))
            if cut_len != 0:
                weight = (cut_len / self.total_records) \
                         * (math.log((cut_len / len(y)) / (class_info[1] / self.total_records), 2))
                weights_per_class[class_info[0]] = (weight, (cut_len / len(y)))
            else:
                weights_per_class[class_info[0]] = (0, 0)
        return weights_per_class

    def _define_interval_for_numeric_feature(self, X, y, attributes_indexes, columns_type):
        """ Define intervals for each numeric feature in the train set.

        Parameters
        ---------
        x: {array-like, sparse matrix}, shape (n_samples, n_features)
            The samples in the train set.

        y: (An array_like object of length n_samples)
            Contains the true class labels for all the samples in X.

        attributes_indexes: (An array_like object of length n_features)
            Contains of the attributes indexes in the train set.

        columns_type: (An array_like object of length n_features)
            Contains the type of each attribute in the train set.
            Each position in the array contain the type of the corresponding attribute in attributes_index array.


        Returns
        --------
            A dictionary contains for each numeric attribute it's data interval.


        """
        intervals_per_attributes = {}

        for attribute_index in attributes_indexes:
            is_continuous = 'category' not in columns_type[attribute_index]
            if is_continuous:
                attribute_data = list(X[:, attribute_index])
                # attribute_data = [round(num, 3) for num in attribute_data]
                self.unique_values_per_attribute[attribute_index] = np.unique(attribute_data)
                data_class_array = list(zip(attribute_data, y))
                data_class_array.sort(key=lambda tup: tup[0])
                intervals_per_attributes[attribute_index] = data_class_array

        return intervals_per_attributes

    def _convert_numeric_values(self, chosen_split_points, chosen_attribute, layer, partial_X):
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
            The index of the attribute upon the network will be splitted by.

        layer: IfnHiddenLayer
            The current HiddenLayer in the network.


        """
        # For the case it's the first layer
        if not bool(chosen_split_points):
            chosen_split_points = self.split_points[chosen_attribute]

        self.unique_values_per_attribute[chosen_attribute] = np.arange(len(chosen_split_points) + 1)
        chosen_split_points.sort()

        if layer is not None:
            # Get the nodes which should be splitted by the chosen attribute
            splitted_nodes = set(self.nodes_splitted_per_attribute[chosen_attribute])
            for node in layer.get_nodes():
                if node in splitted_nodes:
                    partial_x = node.partial_x
                    # convert each value in record[chosen_attribute] to a number between 0 and len(chosen_split_points)
                    for record in partial_x:
                        record[chosen_attribute] = utils.find_split_position(value=record[chosen_attribute],
                                                                             positions=chosen_split_points)
        # First layer
        else:
            # Convert each value in record[chosen_attribute] to a number between 0 and len(chosen_split_points)
            for record in partial_X:
                record[chosen_attribute] = utils.find_split_position(value=record[chosen_attribute],
                                                                     positions=chosen_split_points)

    def _set_terminal_nodes(self, nodes, class_count):
        """ Connecting the given nodes to the terminal nodes in the network.

        Parameters
        ----------
        nodes: (An array_like object of length n)
            Contains a list of objects from the AttributeNode type.

        class_count: (list of length n contain tuples)
            Contain list of tuples - (class value, number of appearances in the train set).


        """

        for node in nodes:
            node.set_terminal()
            # add weight between node and the terminal nodes
            node.set_weight_probability_pair(self._calculate_weights(y=node.partial_y,
                                                                     class_count=class_count))

    def calculate_error_rate(self, X, y):

        return 1 - self.score(X=X, y=y, sample_weight=None)

