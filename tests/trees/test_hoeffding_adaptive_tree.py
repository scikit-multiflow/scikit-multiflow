import numpy as np
from array import array
import os
from skmultiflow.data import ConceptDriftStream, SEAGenerator, HyperplaneGenerator
from skmultiflow.trees import HAT


def test_hat_mc(test_path):
    stream = ConceptDriftStream(stream=SEAGenerator(random_state=1, noise_percentage=0.05),
                                drift_stream=SEAGenerator(random_state=2, classification_function=2,
                                                          noise_percentage=0.05),
                                random_state=1, position=250, width=10)
    stream.prepare_for_use()

    learner = HAT(leaf_prediction='mc')

    cnt = 0
    max_samples = 1000
    y_pred = array('i')
    y_proba = []
    wait_samples = 20

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred.append(learner.predict(X)[0])
            y_proba.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('i', [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                       1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert np.alltrue(y_pred == expected_predictions)

    test_file = os.path.join(test_path, 'test_hoeffding_adaptive_tree_mc.npy')
    data = np.load(test_file)
    assert np.allclose(y_proba, data)

    expected_info = "HAT(binary_split=False, grace_period=200, leaf_prediction='mc',\n" \
                    "    max_byte_size=33554432, memory_estimate_period=1000000, nb_threshold=0,\n" \
                    "    no_preprune=False, nominal_attributes=None, remove_poor_atts=False,\n" \
                    "    split_confidence=1e-07, split_criterion='info_gain',\n" \
                    "    stop_mem_management=False, tie_threshold=0.05)"

    assert learner.get_info() == expected_info

    expected_model_1 = 'Leaf = Class 1.0 | {0.0: 398.0, 1.0: 1000.0}\n'

    assert (learner.get_model_description() == expected_model_1)

    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray

    stream.restart()
    X, y = stream.next_sample(5000)

    learner = HAT(max_byte_size=30, leaf_prediction='mc', grace_period=10)
    learner.partial_fit(X, y)


def test_hat_nb(test_path):
    stream = ConceptDriftStream(stream=SEAGenerator(random_state=1, noise_percentage=0.05),
                                drift_stream=SEAGenerator(random_state=2, classification_function=2,
                                                          noise_percentage=0.05),
                                random_state=1, position=250, width=10)
    stream.prepare_for_use()

    learner = HAT(leaf_prediction='nb')

    cnt = 0
    max_samples = 1000
    y_pred = array('i')
    y_proba = []
    wait_samples = 20

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred.append(learner.predict(X)[0])
            y_proba.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('i', [1, 0, 1, 1, 1, 1, 0, 1, 1, 1,
                                       0, 1, 1, 1, 1, 1, 0, 1, 0, 1,
                                       1, 1, 1, 0, 1, 0, 0, 1, 1, 0,
                                       1, 1, 1, 1, 1, 0, 1, 0, 1, 1,
                                       0, 1, 1, 1, 1, 1, 0, 1, 1])
    assert np.alltrue(y_pred == expected_predictions)

    test_file = os.path.join(test_path, 'test_hoeffding_adaptive_tree_nb.npy')
    data = np.load(test_file)
    assert np.allclose(y_proba, data)

    expected_info = "HAT(binary_split=False, grace_period=200, leaf_prediction='nb',\n" \
                    "    max_byte_size=33554432, memory_estimate_period=1000000, nb_threshold=0,\n" \
                    "    no_preprune=False, nominal_attributes=None, remove_poor_atts=False,\n" \
                    "    split_confidence=1e-07, split_criterion='info_gain',\n" \
                    "    stop_mem_management=False, tie_threshold=0.05)"

    assert learner.get_info() == expected_info
    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray


def test_hat_nba(test_path):
    stream = HyperplaneGenerator(mag_change=0.001, noise_percentage=0.1, random_state=2)

    stream.prepare_for_use()

    learner = HAT(leaf_prediction='nba')

    cnt = 0
    max_samples = 5000
    y_pred = array('i')
    y_proba = []
    wait_samples = 100

    while cnt < max_samples:
        X, y = stream.next_sample()
        # Test every n samples
        if (cnt % wait_samples == 0) and (cnt != 0):
            y_pred.append(learner.predict(X)[0])
            y_proba.append(learner.predict_proba(X)[0])
        learner.partial_fit(X, y)
        cnt += 1

    expected_predictions = array('i', [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0,
                                       1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                                       0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1,
                                       1, 1, 0, 0, 1, 0, 1, 1, 1, 0])

    assert np.alltrue(y_pred == expected_predictions)

    test_file = os.path.join(test_path, 'test_hoeffding_adaptive_tree_nba.npy')
    data = np.load(test_file)
    assert np.allclose(y_proba, data)

    expected_info = "HAT(binary_split=False, grace_period=200, leaf_prediction='nba',\n" \
                    "    max_byte_size=33554432, memory_estimate_period=1000000, nb_threshold=0,\n" \
                    "    no_preprune=False, nominal_attributes=None, remove_poor_atts=False,\n" \
                    "    split_confidence=1e-07, split_criterion='info_gain',\n" \
                    "    stop_mem_management=False, tie_threshold=0.05)"

    assert learner.get_info() == expected_info
    assert type(learner.predict(X)) == np.ndarray
    assert type(learner.predict_proba(X)) == np.ndarray


def test_hoeffding_adaptive_tree_categorical_features(test_path):
    data_path = os.path.join(test_path, 'ht_categorical_features_testcase.npy')
    stream = np.load(data_path)
    # Removes the last two columns (regression targets)
    stream = stream[:, :-2]
    X, y = stream[:, :-1], stream[:, -1]

    nominal_attr_idx = [2, 3, 5, 6]
    learner = HAT(nominal_attributes=nominal_attr_idx)

    learner.partial_fit(X, y, classes=np.unique(y))

    expected_description = "if Attribute 2 = -4.0:\n" \
                           "  Leaf = Class 0 | {0: 556.0, 1: 103.0}\n" \
                           "if Attribute 2 = -3.0:\n" \
                           "  Leaf = Class 0 | {0: 488.0, 1: 156.0}\n" \
                           "if Attribute 2 = -2.0:\n" \
                           "  if Attribute 3 = -3.0:\n" \
                           "    Leaf = Class 0 | {0: 119.0, 1: 15.0}\n" \
                           "  if Attribute 3 = -2.0:\n" \
                           "    Leaf = Class 0 | {0: 97.0, 1: 23.0}\n" \
                           "  if Attribute 3 = -1.0:\n" \
                           "    Leaf = Class 1 | {0: 63.0, 1: 68.0}\n" \
                           "  if Attribute 3 = 0.0:\n" \
                           "    Leaf = Class 1 | {0: 31.0, 1: 71.0}\n" \
                           "if Attribute 2 = -1.0:\n" \
                           "  Leaf = Class 1 | {0: 288.0, 1: 399.0}\n" \
                           "if Attribute 2 = 0.0:\n" \
                           "  Leaf = Class 1 | {0: 178.0, 1: 572.0}\n" \
                           "if Attribute 2 = 1.0:\n" \
                           "  Leaf = Class 1 | {0: 113.0, 1: 552.0}\n"

    assert learner.get_model_description() == expected_description
