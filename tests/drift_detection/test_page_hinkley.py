import os
import numpy as np
from skmultiflow.drift_detection import PageHinkley


def test_page_hinkley(test_path):
    """
    ADWIN drift detection test.
    The first half of the stream contains a sequence corresponding to a normal distribution of integers from 0 to 1.
    From index 999 to 1999 the sequence is a normal distribution of integers from 0 to 7.
    """
    ph = PageHinkley()
    test_file = os.path.join(test_path, 'drift_stream.npy')
    data_stream = np.load(test_file)
    expected_indices = [1013, 1335, 1505, 1758]
    detected_indices = []

    for i in range(data_stream.size):
        ph.add_element(data_stream[i])
        if ph.detected_change():
            detected_indices.append(i)

    assert detected_indices == expected_indices

    expected_info = "PageHinkley(alpha=0.9999, delta=0.005, min_instances=30, threshold=50)"
    assert ph.get_info() == expected_info
