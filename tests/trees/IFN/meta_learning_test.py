from skmultiflow.trees.ifn.meta_learning import MetaLearning
import pytest

alpha = 0.99
number_of_classes = 2
Pe = 0.5
meta_learning = MetaLearning(alpha, number_of_classes)


def test_suite_calculate_initial_window_size():
    test_calculate_initial_window_size(Pe=0.5, expected=0)
    test_calculate_initial_window_size(Pe=0.7, expected=40)

    # invalid input
    with pytest.raises(ValueError):
        test_calculate_initial_window_size(Pe=-1)


def test_calculate_initial_window_size(Pe, expected=None):
    initial_window = meta_learning.calculate_Wint(Pe)
    assert initial_window == expected


def test_suite_calculate_window_size():

    test_calculate_window_size(NI=2, T=0.4, Etr=0.8, expected=19)

    # invalid input
    with pytest.raises(ValueError):
        test_calculate_window_size(NI=1, T=0.4, Etr=0.8)
        test_calculate_window_size(NI=2, T=2, Etr=0.8)
        test_calculate_window_size(NI=2, T=0.4, Etr=2)


def test_calculate_window_size(NI, T, Etr, expected=None):
    window = meta_learning.calculate_new_window(NI, T, Etr)
    assert window == expected


def test_suite_max_diff():
    test_max_diff(Pe=0.7, Etr=0.9, Eval=0.8, add_count=10, expected=0.04245584870124533)

    # invalid input
    with pytest.raises(ValueError):
        test_max_diff(Pe=0.7, Etr=2, Eval=0.8, add_count=10)
        test_max_diff(Pe=0.7, Etr=0.9, Eval=2, add_count=10)
        test_max_diff(Pe=0.7, Etr=0.9, Eval=0.8, add_count=-2)


def test_max_diff(Pe, Etr, Eval, add_count, expected=None):

    meta_learning.calculate_Wint(Pe)
    max_diff = meta_learning.get_max_diff(Etr, Eval, add_count)

    assert max_diff == expected


test_suite_calculate_initial_window_size()
test_suite_calculate_window_size()
test_suite_max_diff()
