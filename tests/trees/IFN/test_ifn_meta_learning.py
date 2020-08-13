from skmultiflow.trees.IFN.IFN_meta_learning import IfnMetaLearning
import pytest

alpha = 0.99
number_of_classes = 2
Pe = 0.5
meta_learning = IfnMetaLearning(alpha, number_of_classes)


def test_suite_calculate_initial_window_size():
    initial_window = meta_learning.calculate_Wint(0.5)
    assert initial_window == 0

    initial_window = meta_learning.calculate_Wint(0.7)
    assert initial_window == 40

    # invalid input
    with pytest.raises(ValueError):
        meta_learning.calculate_Wint(-1)


def test_suite_calculate_window_size():
    window = meta_learning.calculate_new_window(NI=2, T=0.4, Etr=0.8)
    assert window == 19

    # invalid input
    with pytest.raises(ValueError):
        meta_learning.calculate_new_window(NI=1, T=0.4, Etr=0.8)
    with pytest.raises(ValueError):
        meta_learning.calculate_new_window(NI=2, T=2, Etr=0.8)
    with pytest.raises(ValueError):
        meta_learning.calculate_new_window(NI=2, T=0.4, Etr=2)


def test_suite_max_diff():
    meta_learning.calculate_Wint(Pe=0.7)
    max_diff = meta_learning.get_max_diff(Etr=0.9, Eval=0.8, add_count=10)

    assert max_diff == 0.04245584870124533

    meta_learning.calculate_Wint(Pe=0.7)
    # invalid input
    with pytest.raises(ValueError):
        meta_learning.get_max_diff(Etr=2, Eval=0.8, add_count=10)
    with pytest.raises(ValueError):
        meta_learning.get_max_diff(Etr=0.9, Eval=2, add_count=10)
    with pytest.raises(ValueError):
        meta_learning.get_max_diff(Etr=0.9, Eval=0.8, add_count=-2)
