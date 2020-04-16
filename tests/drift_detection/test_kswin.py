from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.drift_detection import KSWIN

def test_kswin_change_detection():
    """
    KSWIN Test

    Content:
    alpha initialisation test.
    alpha has range from (0,1)

    pre obtained data initialisation test.
    data must be list

    KSWIN window size initialisation test.
    0 < stat_size <  w_size

    KSWIN change detector size initialisation test.
    At least 1 false positive must arisie due to the sensitive alpha, when testing the standard
    Sea generator
    """
    try:
        KSWIN(alpha=-0.1)
    except ValueError:
        assert True
    else:
        assert False
    try:
        KSWIN(alpha=1.1)
    except ValueError:
        assert True
    else:
        assert False

    kswin = KSWIN(alpha=0.5)
    assert kswin.alpha == 0.5


    kswin = KSWIN(data="st")
    assert isinstance(kswin.window, list)

    try:
        KSWIN(w_size=-10)
    except ValueError:
        assert True
    else:
        assert False
    try:
        KSWIN(w_size=10, stat_size=30)
    except ValueError:
        assert True
    else:
        assert False

    kswin = KSWIN(alpha=0.001)
    stream = SEAGenerator(classification_function=2,\
     random_state=112, balance_classes=False, noise_percentage=0.28)

    detections, mean = [], []

    for i in range(1000):
        data = stream.next_sample(10)
        batch = data[0][0][0]
        mean.append(batch)
        kswin.add_element(batch)
        if kswin.detected_change():
            mean = []
            detections.append(i)
    assert len(detections) > 1
