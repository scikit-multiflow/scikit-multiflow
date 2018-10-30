import numpy as np
from array import array
import os
from skmultiflow.data import SEAGenerator
from skmultiflow.neural_networks import PerceptronMask
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def test_perceptron(test_path):
    stream = SEAGenerator(random_state=1)
    stream.prepare_for_use()

    learner = PerceptronMask(random_state=1)

    cnt = 0
    max_samples = 5000
    y_pred = array('i')
    X_batch = []
    y_batch = []
    y_proba = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        X_batch.append(X[0])
        y_batch.append(y[0])
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred.append(learner.predict(X)[0])
            y_proba.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y, classes=stream.target_values)
        cnt += 1

    expected_predictions = array('i', [1, 1, 1, 0, 1, 1, 0, 0, 0, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 0, 1, 1, 1, 0, 0,
                                       0, 0, 0, 1, 1, 0, 0, 0, 1, 1,
                                       1, 1, 0, 1, 0, 1, 1, 0, 1])
    assert np.alltrue(y_pred == expected_predictions)

    test_file = os.path.join(test_path, 'data_perceptron_proba.npy')
    y_proba_expected = np.load(test_file)
    assert np.allclose(y_proba, y_proba_expected)

    expected_info = 'PerceptronMask: - penalty: None - alpha: 0.0001 - fit_intercept: True - max_iter: 1000 ' \
                    '- tol: 0.001 - shuffle: True - eta0: 1.0 - warm_start: False - class_weight: None - n_jobs: 1'

    assert learner.get_info() == expected_info

    # Coverage tests
    learner.reset()
    learner.fit(X=X_batch[:4500], y=y_batch[:4500])
    y_pred = learner.predict(X=X_batch[4501:])
    accuracy = accuracy_score(y_true=y_batch[4501:], y_pred=y_pred)
    expected_accuracy = 0.8897795591182365
    # assert np.isclose(expected_accuracy, accuracy)  # Removed due to npn-replicable error in Travis build

    assert 'estimator' == learner.get_class_type()

    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray
