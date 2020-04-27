from skmultiflow.trees.IFN.IOLIN import MetaLearning

alpha = 0.99
number_of_classes = 2
Pe = 0.5


def test_calculate_initial_window_size_default_values():
    meta_learning = MetaLearning(alpha, number_of_classes)
    initial_window = meta_learning.calculate_Wint(Pe)

    expected = 0

    assert initial_window == expected


def test_calculate_initial_window_size():
    Pe = 0.7
    meta_learning = MetaLearning(alpha, number_of_classes)
    initial_window = meta_learning.calculate_Wint(Pe)

    expected = 40

    assert initial_window == expected


def test_calculate_window_size():
    NI = 2
    T = 0.4
    Etr = 0.8

    meta_learning = MetaLearning(alpha, number_of_classes)
    window = meta_learning.calculate_new_window(NI,T,Etr)

    expected = 19

    assert window == expected


def test_max_diff():
    Pe = 0.7
    Etr = 0.9
    Eval = 0.8
    add_count = 10

    meta_learning = MetaLearning(alpha, number_of_classes)
    meta_learning.calculate_Wint(Pe)
    max_diff = meta_learning.get_max_diff(Etr, Eval, add_count)

    expected = 0.04245584870124533

    assert max_diff == expected


test_calculate_initial_window_size_default_values()
test_calculate_initial_window_size()
test_calculate_window_size()
test_max_diff()

