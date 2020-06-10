from skmultiflow.trees import BasicIncremental
from skmultiflow.trees import IfnClassifier
from skmultiflow.data import SEAGenerator
import numpy as np
alpha = 0.99


def test_basic_incremental(tmpdir):
    dir = tmpdir.mkdir("tmpBasicIncremental")
    ifn = IfnClassifier(alpha)
    stream_generator = SEAGenerator(random_state=23)
    basic_incremental = BasicIncremental(ifn, dir, n_min=0, n_max=300, Pe=0.7, data_stream_generator=stream_generator)
    last_model = basic_incremental.generate()

    expected_number_of_generated_models = 4
    number_of_generated_models = basic_incremental.counter - 1

    stream_generator = SEAGenerator(random_state=23)
    X, y = stream_generator.next_sample(basic_incremental.window)
    predictions = basic_incremental.classifier.predict(X)

    expected_predictions = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
                            1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                            1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                            0.0, 1.0, 1.0, 1.0, 0.0]

    correct_predictions = [i for i, j in zip(y, predictions) if i == j]
    expected_correct_predictions = 37

    performance = len(correct_predictions) / len(y)
    expected_performance = 0.925

    assert last_model is not None
    assert number_of_generated_models == expected_number_of_generated_models
    assert np.alltrue(predictions == expected_predictions)
    assert expected_performance == performance
    assert len(correct_predictions) == expected_correct_predictions
