import pickle
import pytest
from sklearn.metrics import accuracy_score
from skmultiflow.data import RandomTreeGenerator
from skmultiflow.trees import IfnClassifier
from sklearn import datasets
import pandas as pd
import filecmp
import numpy as np

alpha = 0.95


def test_iris_dataset():
    columns_type = ['int64', 'int64', 'int64', 'int64']
    clf = IfnClassifier(columns_type=columns_type, alpha=alpha)
    iris = datasets.load_iris()
    X = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
    y = iris.target

    clf.fit(X, y)
    expected_train_accuracy = 0.020000000000000018
    assert np.isclose(expected_train_accuracy, clf.training_error)
    assert len(clf.network.root_node.first_layer.nodes) == 4
    assert len(clf.network.root_node.first_layer.next_layer.nodes) == 5
    assert clf.network.root_node.first_layer.index == 2
    assert clf.network.root_node.first_layer.next_layer.index == 3


def test_model_pickle_stream_dataset(tmpdir):
    columns_type = ['int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64',
                    'int64', 'int64', 'int64', 'int64']
    clf = IfnClassifier(columns_type=columns_type, alpha=alpha)
    stream = RandomTreeGenerator(tree_random_state=112,
                                 sample_random_state=112,
                                 n_num_features=3,
                                 n_categories_per_cat_feature=2)
    x_train, y_train = stream.next_sample(100)
    print(x_train)
    x_test, y_test = stream.next_sample(100)
    X_train_df = pd.DataFrame(x_train)

    clf.fit(X_train_df, y_train)
    pickle_file = tmpdir.join("clf.pickle")
    pickle.dump(clf, open(pickle_file, "wb"))
    network_structure_file = tmpdir.join("network_structure.txt")
    clf.network.create_network_structure_file(path=network_structure_file)
    y_pred = clf.predict(x_test)

    loaded_clf = pickle.load(open(pickle_file, "rb"))
    pickle_network_structure_file = tmpdir.join("loaded_network_structure.txt")
    loaded_clf.network.create_network_structure_file(path=pickle_network_structure_file)
    loaded_y_pred = loaded_clf.predict(x_test)

    assert filecmp.cmp(pickle_network_structure_file, network_structure_file) is True
    assert np.array_equal(y_pred, loaded_y_pred)


def test_partial_fit():
    columns_type = ['int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64', 'int64',
                    'int64', 'int64', 'int64', 'int64']
    stream = RandomTreeGenerator(tree_random_state=112,
                                 sample_random_state=112,
                                 n_num_features=3,
                                 n_categories_per_cat_feature=2)

    estimator = IfnClassifier(columns_type, alpha, window_size=100)

    X, y = stream.next_sample(100)
    estimator.partial_fit(X, y)

    cnt = 0
    max_samples = 2000
    predictions = []
    true_labels = []
    wait_samples = 100
    correct_predictions = 0

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(estimator.predict(X)[0])
            true_labels.append(y[0])
            if np.array_equal(y[0], predictions[-1]):
                correct_predictions += 1

        estimator.partial_fit(X, y)
        cnt += 1

    performance = correct_predictions / len(predictions)
    expected_predictions = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]

    expected_correct_predictions = 14
    expected_performance = 0.7368421052631579

    assert np.alltrue(predictions == expected_predictions)
    assert np.isclose(expected_performance, performance)
    assert correct_predictions == expected_correct_predictions
