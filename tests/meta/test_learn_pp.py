from skmultiflow.data import RandomTreeGenerator
from skmultiflow.meta.learn_pp import LearnPPClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def test_learn_pp():
    stream = RandomTreeGenerator(tree_random_state=2212, sample_random_state=2212)

    estimator = DecisionTreeClassifier(random_state=2212)
    classifier = LearnPPClassifier(base_estimator=estimator, n_estimators=5, n_ensembles=5, random_state=2212)

    m = 200

    # Keeping track of sample count and correct prediction count
    sample_count = 0
    corrects = 0

    # Pre training the classifier with 200 samples
    X, y = stream.next_sample(m)
    classifier.partial_fit(X, y, classes=stream.target_values)
    predictions = []

    for i in range(10):
        X, y = stream.next_sample(200)
        pred = classifier.predict(X)
        classifier.partial_fit(X, y)

        if pred is not None:
            corrects += np.sum(y == pred)
            predictions.append(pred[0])
        sample_count += m

    acc = corrects / sample_count

    expected_correct_predictions = 1138
    expected_acc = 0.569
    expected_predictions = [0, 1, 0, 0, 1, 1, 0, 0, 0, 0]

    assert np.alltrue(predictions == expected_predictions)
    assert np.isclose(expected_acc, acc)
    assert corrects == expected_correct_predictions
    assert type(classifier.predict(X)) == np.ndarray

    expected_info = "LearnPPClassifier(base_estimator=DecisionTreeClassifier(ccp_alpha=0.0, " \
                    "class_weight=None, criterion='gini', max_depth=None, max_features=None, max_leaf_nodes=None, " \
                    "min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, " \
                    "min_samples_split=2, min_weight_fraction_leaf=0.0, presort='deprecated', " \
                    "random_state=2212, splitter='best'), error_threshold=0.5, n_ensembles=5, " \
                    "n_estimators=5, random_state=2212, window_size=100)"
    info = " ".join([line.strip() for line in classifier.get_info().split()])
    assert info == expected_info

    # For coverage purposes
    classifier.reset()


def test_learn_pp_early_stop():
    # Corner case where all observations belong to the same class:
    # not all ensemble members need to be trained (PR #223)
    stream = RandomTreeGenerator(
        tree_random_state=7, sample_random_state=8, n_classes=1
    )

    estimator = DecisionTreeClassifier(random_state=42)
    classifier = LearnPPClassifier(
        base_estimator=estimator, n_estimators=5, n_ensembles=5,
        random_state=7
    )

    m = 200

    # Keeping track of sample count and correct prediction count
    sample_count = 0
    corrects = 0

    # Pre training the classifier with 200 samples
    X, y = stream.next_sample(m)
    classifier.partial_fit(X, y, classes=stream.target_values)
    predictions = []

    for i in range(5):
        X, y = stream.next_sample(m)
        pred = classifier.predict(X)
        classifier.partial_fit(X, y)

        if pred is not None:
            corrects += np.sum(y == pred)
            predictions.append(pred[0])
        sample_count += m

    acc = corrects / sample_count

    expected_correct_predictions = 1000
    expected_acc = 1.0
    expected_predictions = [0, 0, 0, 0, 0]

    assert np.alltrue(predictions == expected_predictions)
    assert np.isclose(expected_acc, acc)
    assert corrects == expected_correct_predictions
