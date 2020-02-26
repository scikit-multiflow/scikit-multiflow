from skmultiflow.data import ConceptDriftStream
from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta import StreamingRandomPatchesClassifier

import numpy as np


def test_srp_randompatches():

    stream = ConceptDriftStream(position=1000, width=20, random_state=1)
    learner = StreamingRandomPatchesClassifier(base_estimator=NaiveBayes(),
                                               n_estimators=3,
                                               subspace_mode='percentage',
                                               training_method='randompatches',
                                               random_state=1)

    y_expected = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                             0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                             0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int)

    run_prequential_supervised(stream, learner, max_samples=2000, n_wait=40, y_expected=y_expected)


def test_srp_randomsubspaces():

    stream = ConceptDriftStream(position=1000, width=20, random_state=1)
    learner = StreamingRandomPatchesClassifier(base_estimator=NaiveBayes(),
                                               n_estimators=3,
                                               subspace_mode='percentage',
                                               training_method='randomsubspaces',
                                               random_state=1)

    y_expected = np.asarray([0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                             0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             1, 0, 1, 1, 1, 0, 1, 1, 0, 1,
                             0, 0, 1, 0, 1, 1, 0, 0, 0, 0], dtype=np.int)

    run_prequential_supervised(stream, learner, max_samples=2000, n_wait=40, y_expected=y_expected)


def test_srp_resampling():
    stream = ConceptDriftStream(position=1000, width=20, random_state=1)
    learner = StreamingRandomPatchesClassifier(base_estimator=NaiveBayes(),
                                               n_estimators=3,
                                               subspace_mode='percentage',
                                               training_method='resampling',
                                               random_state=1)

    y_expected = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                             0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                             1, 1, 1, 0, 1, 0, 0, 1, 0, 1,
                             0, 0, 0, 0, 0, 0, 0, 1, 1, 0], dtype=np.int)

    run_prequential_supervised(stream, learner, max_samples=2000, n_wait=40, y_expected=y_expected)


def run_prequential_supervised(stream, learner, max_samples, n_wait, y_expected=None):
    y_pred = np.zeros(max_samples // n_wait, dtype=np.int)
    y_true = np.zeros(max_samples // n_wait, dtype=np.int)
    j = 0

    for i in range(max_samples):
        X, y = stream.next_sample()
        # Test every n samples
        if (i % n_wait == 0) and (i != 0):
            y_pred[j] = int(learner.predict(X)[0])
            y_true[j] = (y[0])
            j += 1
        learner.partial_fit(X, y)

    assert type(learner.predict(X)) == np.ndarray

    if y_expected is not None:
        assert np.alltrue(y_pred == y_expected)
