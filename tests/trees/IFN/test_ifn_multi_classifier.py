import pickle
from sklearn.metrics import accuracy_score
from skmultiflow.trees import IfnClassifierMulti
from skmultiflow.trees.ifn.data_processing_multi import DataProcessorMulti
import pytest
import filecmp
import numpy as np

dataset_path = "tests/trees/IFN/Chess_multi.csv"
test_size_percentage = 0.3
alpha = 0.99

# def test_classifier_const_dataset(multi_label=False):
#     _setup_test_env()
#     clf = IfnClassifierMulti(alpha, multi_label)
#     dp = DataProcessor()
#     x_train, x_test, y_train, y_test = dp.convert(dataset_path, test_size_percentage)
#
#     clf.fit(x_train, y_train)
#     clf.network.create_network_structure_file()
#     y_pred = clf.predict(x_test)
#
#     assert isinstance(y_pred, pd.DataFrame)
#     assert accuracy_score(y_test, y_pred) > 0.2
#
#     _clean_test_env()


def test__model_pickle_const_dataset_multi_target(tmpdir):
    clf = IfnClassifierMulti(alpha, multi_label=False)
    dp = DataProcessorMulti()
    x_train, x_test, y_train, y_test = dp.convert(dataset_path, test_size_percentage)

    clf.fit(x_train, y_train)
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
    print("accuracy:", accuracy_score(y_test, y_pred))
    assert accuracy_score(y_test, y_pred) == accuracy_score(y_test, loaded_y_pred)


def test__model_pickle_const_dataset_multi_label(tmpdir):
    clf = IfnClassifierMulti(alpha, multi_label=False)
    dp = DataProcessorMulti()
    x_train, x_test, y_train, y_test = dp.convert(dataset_path, test_size_percentage)

    clf.fit(x_train, y_train)
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
    print("accuracy:", accuracy_score(y_test, y_pred))
    assert accuracy_score(y_test, y_pred) == accuracy_score(y_test, loaded_y_pred)
