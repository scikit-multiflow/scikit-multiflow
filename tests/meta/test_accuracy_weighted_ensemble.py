from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
from skmultiflow.data.concept_drift_stream import ConceptDriftStream
from skmultiflow.data.file_stream import FileStream
from skmultiflow.meta import AccuracyWeightedEnsemble, AccuracyWeightedEnsembleMOA
from sklearn.tree import DecisionTreeClassifier
from skmultiflow.trees import HoeffdingTree
from sklearn.naive_bayes import GaussianNB
from skmultiflow.bayes import NaiveBayes
from skmultiflow.evaluation import EvaluatePrequential
import numpy as np


def test_awe():
    # prepare the stream
    stream = HyperplaneGenerator(random_state=420)
    stream.prepare_for_use()

    # prepare the ensemble
    classifier = AccuracyWeightedEnsemble(n_estimators=10,
                                          base_estimator=GaussianNB(),
                                          window_size=200, n_splits=5)

    # test reset
    classifier.reset()
    assert classifier.n_estimators == 10
    assert classifier.window_size == 200
    assert classifier.n_splits == 5

    # test the classifier
    m = 500
    # Pre training the classifier
    X, y = stream.next_sample(m)
    classifier.partial_fit(X, y, classes=stream.target_values)

    # Keeping track of sample count and correct prediction count
    sample_count = 0
    corrects = 0

    for i in range(10):
        X, y = stream.next_sample(m)
        pred = classifier.predict(X)
        classifier.partial_fit(X, y)

        if pred is not None:
            corrects += np.sum(y == pred)
        sample_count += m

    acc = corrects / sample_count
    print(corrects, sample_count, acc)


def test_performance_awe(dataset, base_estimator=NaiveBayes(), n_wait=1000):
    # prepare the stream
    stream = FileStream(filepath=dataset)
    stream.prepare_for_use()

    # prepare the classifier
    classifier = AccuracyWeightedEnsemble(n_estimators=10, n_kept_estimators=30,
                                          base_estimator=base_estimator,
                                          window_size=200, n_splits=5)

    # prepare the evaluator
    evaluator = EvaluatePrequential(max_samples=100000, batch_size=1, pretrain_size=0,
                                    metrics=["accuracy", "kappa"], show_plot=False, restart_stream=True,
                                    output_file="D:/Study/M2_DK/Data_Stream/result/result.csv", n_wait=n_wait)

    # run stuffs
    evaluator.evaluate(stream, classifier)
    print("\nDATASET:", dataset)
    print("BASE LEARNER:", base_estimator.get_info())
    print("N_WAIT:", n_wait)


# test_performance_awe(dataset="agr_a_10k.arff.csv", n_wait=500)
# test_performance_awe(dataset="elec.arff.csv", n_wait=500)
# test_performance_awe(dataset="hyperplaneData.csv", n_wait=1000)
# test_performance_awe(dataset="conceptDriftData100K.csv", n_wait=1000)
# test_performance_awe(dataset="covtype.csv", n_wait=1000)
# test_performance_awe(dataset="randomTree100K.arff.csv", n_wait=1000)
# test_performance_awe(dataset="elec.arff.csv", base_estimator=NaiveBayes(), n_wait=1000)
test_performance_awe(dataset="elec.arff.csv", base_estimator=HoeffdingTree(), n_wait=1000)