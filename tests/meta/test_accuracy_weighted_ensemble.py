from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
from skmultiflow.data.file_stream import FileStream
from skmultiflow.meta import AccuracyWeightedEnsemble
from skmultiflow.bayes import NaiveBayes
from skmultiflow.evaluation import EvaluatePrequential
import numpy as np
from array import array


def test_awe():
    # prepare the stream
    stream = HyperplaneGenerator(random_state=1)
    stream.prepare_for_use()

    # prepare the ensemble
    classifier = AccuracyWeightedEnsemble(n_estimators=5, n_kept_estimators=10,
                                          base_estimator=NaiveBayes(),
                                          window_size=200, n_splits=5)

    # test the classifier
    max_samples = 5000
    cnt = 0
    wait_samples = 100
    predictions = array('i')
    correct = 0
    while cnt < max_samples:
        X, y = stream.next_sample()
        pred = classifier.predict(X)
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            predictions.append(int(pred[0]))
        classifier.partial_fit(X, y)
        cnt += 1
        if pred[0] == y:
            correct += 1

    # assert model predictions
    expected_predictions = array('i', [
        0, 0, 1, 1, 0, 0, 1, 0, 1, 0,
        0, 0, 0, 1, 0, 1, 0, 1, 1, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 1, 0, 0, 1, 0, 1, 1, 1, 0,
        1, 1, 1, 0, 0, 1, 1, 1, 1
    ])

    # assert model performance
    expected_accuracy = 0.875
    accuracy = correct / max_samples
    assert expected_accuracy == accuracy

    assert np.alltrue(predictions == expected_predictions)

    # assert model information
    expected_info = "AccuracyWeightedEnsemble: n_estimators: 5 - " \
                    "n_kept_estimators: 10 - " \
                    "base_estimator: NaiveBayes: nominal attributes: [] -  - " \
                    "window_size: 200 - " \
                    "n_splits: 5"
    assert classifier.get_info() == expected_info


def test_performance_awe(dataset=None, base_estimator=NaiveBayes(), n_wait=1000):
    # prepare the stream
    # stream = FileStream(filepath=dataset)
    stream = HyperplaneGenerator()
    stream.prepare_for_use()

    # prepare the classifier
    classifier = AccuracyWeightedEnsemble(n_estimators=10, n_kept_estimators=30,
                                          base_estimator=base_estimator,
                                          window_size=200, n_splits=5)

    # prepare the evaluator
    evaluator = EvaluatePrequential(max_samples=50000, batch_size=1, pretrain_size=0,
                                    metrics=["accuracy", "kappa"], show_plot=False, restart_stream=True,
                                    n_wait=n_wait)

    # run stuffs
    evaluator.evaluate(stream, classifier)
    print("\nDATASET:", dataset)
    print("BASE LEARNER:", base_estimator.get_info())
    print("N_WAIT:", n_wait)
