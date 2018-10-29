import os
import numpy as np
from array import array
from sklearn.metrics import mean_absolute_error
from skmultiflow.data import RegressionGenerator
from skmultiflow.trees import RegressionHoeffdingTree


def test_hoeffding_tree():
    stream = RegressionGenerator(n_samples=500, n_features=20, n_informative=15, random_state=1)
    stream.prepare_for_use()

    learner = RegressionHoeffdingTree(leaf_prediction='mean')

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

    expected_predictions = array('d', [102.38946041769101, 55.6584574987656, 5.746076599168373, 17.11797209372667,
                                       2.566888222752787, 9.188247802192826, 17.87894804676911, 15.940629626883966,
                                       8.981172175448485, 13.152624115190092, 11.106058099429399, 6.473195313058236,
                                       4.723621479590173, 13.825568609556493, 8.698873073880696, 1.6452441811010252,
                                       5.123496188584294, 6.34387187194982, 5.9977733790395105, 6.874251577667707,
                                       4.605348088338317, 8.20112636572672, 9.032631648758098, 4.428189978974459,
                                       4.249801041367518, 9.983272668044492, 12.859518508979734, 11.741395774380285,
                                       11.230028410261868, 9.126921979081521, 9.132146661688296, 7.750655625124709,
                                       6.445145118245414, 5.760928671876355, 4.041291302080659, 3.591837600560529,
                                       0.7640424010500604, 0.1738639840537784, 2.2068337802212286, -81.05302946841077,
                                       96.17757415335177, -77.35894903819677, 95.85568683733698, 99.1981674250886,
                                       99.89327888035015, 101.66673013734784, -79.1904234513751, -80.42952143783687,
                                       100.63954789983896])
    assert np.allclose(y_pred, expected_predictions)

    error = mean_absolute_error(y_true, y_pred)
    expected_error = 143.11351404083086
    assert np.isclose(error, expected_error)

    expected_info = 'RegressionHoeffdingTree: max_byte_size: 33554432 - memory_estimate_period: 1000000 ' \
                    '- grace_period: 200 - split_criterion: variance reduction - split_confidence: 1e-07 ' \
                    '- tie_threshold: 0.05 - binary_split: False - stop_mem_management: False ' \
                    '- remove_poor_atts: False - no_pre_prune: False - leaf_prediction: mean - nb_threshold: 0 ' \
                    '- nominal_attributes: [] - '
    assert learner.get_info() == expected_info

    assert isinstance(learner.get_model_description(), type(''))
    assert type(learner.predict(X)) == np.ndarray


def test_hoeffding_tree_perceptron():
    stream = RegressionGenerator(n_samples=500, n_features=20, n_informative=15, random_state=1)
    stream.prepare_for_use()

    learner = RegressionHoeffdingTree(leaf_prediction='perceptron', random_state=1)

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

    expected_predictions = array('d', [1198.4326121743168, 456.36607750881586, 927.9912160545144, 1160.4797981899128,
                                       506.50541829176535, -687.8187227095925, -677.8120094065415, 231.14888704761225,
                                       -284.46324039942937, -255.69195985557175, 47.58787439365423, -135.22494016284043,
                                       -10.351457437330152, 164.95903200643997, 360.72854984472383, 193.30633911830088,
                                       -64.23638301570358, 587.9771578214296, 649.8395655757931, 481.01214222804026,
                                       305.4402728117724, 266.2096493865043, -445.11447171009775, -567.5748694154349,
                                       -68.70070048021438, -446.79910655850153, -115.892348067663, -98.26862866231015,
                                       71.04707905920286, -10.239274802165584, 18.748731569441812, 4.971217265129857,
                                       172.2223575990573, -655.2864976783711, -129.69921313686626, -114.01187375876822,
                                       -405.66166686550963, -215.1264381928009, -345.91020370426247, -80.49330468453074,
                                       108.78958382083302, 134.95267043280126, -398.5273538477553, -157.1784910649728,
                                       219.72541225645654, -100.91598162899217, 80.9768574308987, -296.8856956382453,
                                       251.9332271253148])
    assert np.allclose(y_pred, expected_predictions)

    error = mean_absolute_error(y_true, y_pred)
    expected_error = 362.98595964244623
    assert np.isclose(error, expected_error)

    expected_info = 'RegressionHoeffdingTree: max_byte_size: 33554432 - memory_estimate_period: 1000000 ' \
                    '- grace_period: 200 - split_criterion: variance reduction - split_confidence: 1e-07 ' \
                    '- tie_threshold: 0.05 - binary_split: False - stop_mem_management: False ' \
                    '- remove_poor_atts: False - no_pre_prune: False - leaf_prediction: perceptron - nb_threshold: 0 ' \
                    '- nominal_attributes: [] - '
    assert learner.get_info() == expected_info

    assert isinstance(learner.get_model_description(), type(''))
    assert type(learner.predict(X)) == np.ndarray


def test_hoeffding_tree_coverage(test_path):
    # Cover nominal attribute observer
    test_file = os.path.join(test_path, 'regression_data.npz')
    data = np.load(test_file)
    X = data['X']
    y = data['y']

    learner = RegressionHoeffdingTree(leaf_prediction='mean', nominal_attributes=[i for i in range(3)])
    learner.partial_fit(X, y)
