import os
import filecmp
import difflib
import numpy as np
from skmultiflow.data import RandomTreeGenerator
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.evaluation import EvaluateInfluential
from skmultiflow.data import InfluentialStream

tmpdir = "/Users/tinekejelsma/scikit-multiflow/tests/evaluation/"


def test_evaluate_distribution_table(tmpdir, test_path):
    # Setup file stream
    stream = InfluentialStream(self_defeating=1, self_fulfilling=1)
    print("feature names: ", stream.feature_names)

    # Setup learner
    nominal_attr_idx = [x for x in range(15, len(stream.feature_names))]
    learner = HoeffdingTreeClassifier(nominal_attributes=nominal_attr_idx)

    # Setup evaluator
    max_samples = 1000
    metrics = ['accuracy', 'kappa', 'kappa_t']
    output_file = os.path.join(tmpdir, "influential_summary.csv")
    evaluator = EvaluateInfluential(max_samples=max_samples,
                                    metrics=metrics,
                                    output_file=output_file,
                                    batch_size=1,
                                    n_time_windows=2,
                                    n_intervals=4)

    # Evaluate
    result = evaluator.evaluate(stream=stream, model=[learner])
    result_learner = result[0]

    assert isinstance(result_learner, HoeffdingTreeClassifier)

    assert learner.get_model_measurements == result_learner.get_model_measurements

    # check if the time windows are correct
    assert len(evaluator.distribution_table) == 2
    # check whether the numbers of features are corrects
    assert len(evaluator.distribution_table[1]) == len(stream.feature_names)
    # check whether the intervals are correct
    assert len(evaluator.distribution_table[1][1]) == 4

    expected_file = os.path.join(test_path, 'influential_summary.csv')
    compare_files(output_file, expected_file)


def test_evaluate_regression_coverage(tmpdir):
    # A simple coverage test. Tests for metrics are placed in the corresponding test module.
    from skmultiflow.trees import HoeffdingTreeRegressor

    max_samples = 1000

    # Stream
    stream = InfluentialStream(self_defeating=1, self_fulfilling=1)

    # Learner
    htr = HoeffdingTreeRegressor()

    output_file = os.path.join(str(tmpdir), "influential_summary.csv")
    metrics = ['mean_square_error', 'mean_absolute_error']
    evaluator = EvaluateInfluential(max_samples=max_samples,
                                    metrics=metrics,
                                    output_file=output_file,
                                    batch_size=1,
                                    n_time_windows=2,
                                    n_intervals=4)

    evaluator.evaluate(stream=stream, model=htr, model_names=['HTR'])


def test_evaluate_coverage(tmpdir):
    from skmultiflow.bayes import NaiveBayes

    max_samples = 1000

    # Stream
    stream = InfluentialStream(self_defeating=1, self_fulfilling=1)

    # Learner
    nb = NaiveBayes()

    output_file = os.path.join(str(tmpdir), "influential_summary.csv")
    metrics = ['running_time', 'model_size']
    evaluator = EvaluateInfluential(max_samples=max_samples,
                                    metrics=metrics,
                                    data_points_for_classification=True,
                                    output_file=output_file,
                                    batch_size=1,
                                    n_time_windows=2,
                                    n_intervals=4)

    evaluator.evaluate(stream=stream, model=nb, model_names=['NB'])


def compare_files(test, expected):
    lines_expected = open(expected).readlines()
    lines_test = open(test).readlines()

    print(''.join(difflib.ndiff(lines_test, lines_expected)))
    filecmp.clear_cache()
    assert filecmp.cmp(test, expected) is True
