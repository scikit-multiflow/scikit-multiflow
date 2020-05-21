import pickle
from sklearn.metrics import accuracy_score
from skmultiflow.trees import IfnClassifier
from skmultiflow.trees.ifn._data_processing import DataProcessor
import os
import filecmp
import numpy as np
import shutil

dataset_path = "C:\\Users\איתן אביטן\PycharmProjects\scikit-multiflow-IFN\skml\\tests\datasets/credit.csv"

test_size_percentage = 0.3
alpha = 0.99
test_tmp_folder = "tmp"


def _clean_test_env():
    shutil.rmtree(test_tmp_folder, ignore_errors=True)


def _setup_test_env():
    os.mkdir(test_tmp_folder)


def test_classifier_const_dataset():
    _setup_test_env()
    clf = IfnClassifier(alpha)
    dp = DataProcessor()
    x_train, x_test, y_train, y_test = dp.convert(dataset_path, test_size_percentage)

    clf.fit(x_train, y_train)
    clf.network.create_network_structure_file()
    y_pred = clf.predict(x_test)

    expected_pred = np.array([1, 2, 3, 4, 5])  # maybe change to get from file

    assert filecmp.cmp('tmp/network_structure.txt', 'expert_network_structure.txt') is True
    assert np.array_equal(y_pred, expected_pred)
    assert accuracy_score(y_test, y_pred) == accuracy_score(y_test, expected_pred)

    _clean_test_env()


def test__model_pickle_const_dataset():
    # try:
    _setup_test_env()
    clf = IfnClassifier(alpha)
    dp = DataProcessor()
    x_train, x_test, y_train, y_test = dp.convert(dataset_path, test_size_percentage)

    clf.fit(x_train, y_train)
    pickle.dump(clf, open("tmp/clf.pickle", "wb"))
    clf.network.create_network_structure_file()
    os.rename("tmp/network_structure.txt", "tmp/clf_network_structure.txt")
    y_pred = clf.predict(x_test)

    loaded_clf = pickle.load(open("tmp/clf.pickle", "rb"))
    loaded_clf.network.create_network_structure_file()
    os.rename("tmp/network_structure.txt", "tmp/loaded_clf_network_structure.txt")
    loaded_y_pred = loaded_clf.predict(x_test)

    assert filecmp.cmp('tmp/loaded_clf_network_structure.txt', 'tmp/clf_network_structure.txt') is True
    assert np.array_equal(y_pred, loaded_y_pred)
    print(accuracy_score(y_test, y_pred))
    assert accuracy_score(y_test, y_pred) == accuracy_score(y_test, loaded_y_pred)

    _clean_test_env()


test__model_pickle_const_dataset()
