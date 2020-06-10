from skmultiflow.data import SEAGenerator
from skmultiflow.trees import OnlineNetworkRegenerative
from src.skmultiflow.trees import IfnClassifier
import numpy as np

alpha = 0.99


def test_regenerative(tmpdir):
    dir = tmpdir.mkdir("tmpOLIN")
    ifn = IfnClassifier(alpha)
    stream_generator = SEAGenerator(random_state=23)
    regenerative = OnlineNetworkRegenerative(ifn, dir, n_min=0, n_max=300, Pe=0.7,
                                             data_stream_generator=stream_generator)
    last_model = regenerative.generate()

    expected_number_of_generated_models = 2
    number_of_generated_models = regenerative.counter - 1

    stream_generator = SEAGenerator(random_state=23)
    X, y = stream_generator.next_sample(60)
    predictions = regenerative.classifier.predict(X)
    print(predictions)
    expected_predictions = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                            1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0,
                            1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
                            1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                            1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
                            1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
                            1.0, 0.0, 1.0, 1.0]

    correct_predictions = [i for i, j in zip(y, predictions) if i == j]
    expected_correct_predictions = 58

    performance = len(correct_predictions) / len(y)
    expected_performance =  0.9666666666666667

    assert last_model is not None
    assert number_of_generated_models == expected_number_of_generated_models
    assert np.alltrue(predictions == expected_predictions)
    assert expected_performance == performance
    assert len(correct_predictions) == expected_correct_predictions
