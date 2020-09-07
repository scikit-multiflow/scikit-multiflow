import os
import numpy as np
import pytest

from skmultiflow.data.generator.concept_drift_stream_generator import ConceptDriftStreamGenerator


def test_concept_drift_stream(test_path):
    stream = ConceptDriftStreamGenerator(random_state=1, position=20, width=5)

    # Load test data corresponding to first 30 instances
    test_file = os.path.join(test_path, 'concept_drift_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    for j in range(0,30):
        X, y = stream.next_sample()
        assert np.alltrue(np.isclose(X, X_expected[j]))
        assert np.alltrue(np.isclose(y[0], y_expected[j]))

    expected_info = "ConceptDriftStreamGenerator(alpha=None, " \
                    "drift_stream=AGRAWALGenerator(balance_classes=False, classification_function=2, " \
                    "perturbation=0.0, random_state=112), " \
                    "position=20, random_state=1, " \
                    "stream=AGRAWALGenerator(balance_classes=False, classification_function=0, " \
                    "perturbation=0.0, random_state=112), width=5)"
    assert stream.get_info() == expected_info


def test_concept_drift_stream_with_alpha(test_path):
    stream = ConceptDriftStreamGenerator(alpha=0.01, random_state=1, position=20)

    expected_info = "ConceptDriftStreamGenerator(alpha=0.01, " \
                    "drift_stream=AGRAWALGenerator(balance_classes=False, classification_function=2, " \
                    "perturbation=0.0, random_state=112), " \
                    "position=20, random_state=1, " \
                    "stream=AGRAWALGenerator(balance_classes=False, classification_function=0, " \
                    "perturbation=0.0, random_state=112), width=5729)"

    assert stream.get_info() == expected_info

    with pytest.warns(FutureWarning) as actual_warning:
        ConceptDriftStreamGenerator(alpha=0, random_state=1, position=20)

    assert actual_warning[0].message.args[0] == "Default value for 'alpha' has changed from 0 " \
                                            "to None. 'alpha=0' will throw an error from v0.7.0"

