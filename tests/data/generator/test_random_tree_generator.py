import os
import numpy as np
from skmultiflow.data.generator.random_tree_generator import RandomTreeGenerator


def test_random_tree_generator(test_path):
    stream = RandomTreeGenerator(tree_random_state=23, sample_random_state=12, n_classes=2, n_cat_features=2,
                                 n_num_features=5, n_categories_per_cat_feature=5, max_tree_depth=6, min_leaf_depth=3,
                                 fraction_leaves_per_level=0.15)

    # Load test data corresponding to first 10 instances
    test_file = os.path.join(test_path, 'random_tree_stream.npz')
    data = np.load(test_file)
    X_expected = data['X']
    y_expected = data['y']

    for j in range(0,10):
        X, y = stream.next_sample()
        assert np.alltrue(np.isclose(X, X_expected[j]))
        assert np.alltrue(np.isclose(y[0], y_expected[j]))

    expected_info = "RandomTreeGenerator(fraction_leaves_per_level=0.15, max_tree_depth=6," \
                    "min_leaf_depth=3, n_cat_features=2, n_categories_per_cat_feature=5, n_classes=2," \
                    "n_num_features=5, sample_random_state=12, tree_random_state=23)"
    assert stream.get_info() == expected_info
