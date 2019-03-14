from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
from skmultiflow.meta import AccuracyWeightedEnsemble
from sklearn.tree import DecisionTreeClassifier

def test_awe():
    # prepare the stream
    stream = HyperplaneGenerator()
    stream.prepare_for_use()

    # prepare the ensemble
    clf = AccuracyWeightedEnsemble()

    # test reset
    clf.reset()
    assert clf.n_estimators == 10
    assert clf.base_estimator == DecisionTreeClassifier()
    assert clf.window_size == 200
    assert clf.n_splits == 5
