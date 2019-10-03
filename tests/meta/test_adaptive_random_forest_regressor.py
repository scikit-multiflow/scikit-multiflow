import os
import numpy as np
from array import array
from sklearn.metrics import mean_absolute_error
from skmultiflow.data import RegressionGenerator
from skmultiflow.meta import AdaptiveRandomForestRegressor


def test_adaptive_random_forest_regressor():
    stream = RegressionGenerator(n_samples=500, n_features=20, n_informative=15, random_state=1)
    stream.prepare_for_use()

    learner = AdaptiveRandomForestRegressor(random_state=1)

    cnt = 0
    max_samples = 500
    y_pred = array('d')
    y_true = array('d')
    wait_samples = 10

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred.append(learner.predict(X)[0])
            y_true.append(y[0])
        learner.partial_fit(X, y)
        cnt += 1

    error = mean_absolute_error(y_true, y_pred)
    expected_error = 155.5787786499754
    assert np.isclose(error, expected_error)

    assert type(learner.predict(X)) == np.ndarray

    expected_info = "AdaptiveRandomForestRegressor(binary_split=False, disable_weighted_vote=False,\n" \
                    "                              drift_detection_method=ADWIN(delta=0.001),\n" \
                    "                              grace_period=200, lambda_value=6,\n" \
                    "                              leaf_prediction='perceptron',\n" \
                    "                              learning_ratio_const=True,\n" \
                    "                              learning_ratio_decay=0.001,\n" \
                    "                              learning_ratio_perceptron=0.02,\n" \
                    "                              max_byte_size=33554432, max_features=4,\n" \
                    "                              memory_estimate_period=1000000, n_estimators=10,\n" \
                    "                              nb_threshold=0, no_preprune=False,\n" \
                    "                              nominal_attributes=None, random_state=1,\n" \
                    "                              remove_poor_atts=False, split_confidence=1e-07,\n" \
                    "                              stop_mem_management=False, tie_threshold=0.05,\n" \
                    "                              warning_detection_method=ADWIN(delta=0.01))"
    print(learner.get_info())

    assert learner.get_info() == expected_info
