from skmultiflow.data.generator.concept_drift_stream_generator import ConceptDriftStreamGenerator
from skmultiflow.data.generator.sea_generator import SEAGenerator
from skmultiflow.lazy import KNNADWINClassifier
from array import array
import numpy as np


def test_knn_adwin():
    stream = ConceptDriftStreamGenerator(stream=SEAGenerator(random_state=1),
                                drift_stream=SEAGenerator(random_state=2, classification_function=2),
                                random_state=1, position=250, width=10)

    learner = KNNADWINClassifier(n_neighbors=8, leaf_size=40, max_window_size=200)

    cnt = 0
    max_samples = 1000
    predictions = array('i')
    correct_predictions = 0
    wait_samples = 20

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(learner.predict(X)[0])
            if y[0] == predictions[-1]:
                correct_predictions += 1
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('i', [1, 0, 1, 1, 1, 1, 0, 1, 1, 1,
                                       0, 1, 1, 0, 1, 1, 0, 1, 0, 1,
                                       1, 1, 1, 0, 1, 0, 0, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       0, 1, 1, 1, 0, 1, 0, 1, 1])
    assert np.alltrue(predictions == expected_predictions)

    expected_correct_predictions = 46
    assert correct_predictions == expected_correct_predictions

    learner.reset()
    assert learner.data_window.size == 0

    expected_info = "KNNADWINClassifier(leaf_size=40, max_window_size=200, " \
                    "metric='euclidean', n_neighbors=8)"
    info = " ".join([line.strip() for line in learner.get_info().split()])
    assert info == expected_info

    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray
