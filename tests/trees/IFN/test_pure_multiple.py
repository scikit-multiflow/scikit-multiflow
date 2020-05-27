from skmultiflow.trees import PureMultiple
from skmultiflow.trees import IfnClassifier

alpha = 0.99


def test_pure_multiple(tmpdir):
    dir = tmpdir.mkdir("tmpPureMultiple")
    ifn = IfnClassifier(alpha)
    pure_IOLIN = PureMultiple(ifn, dir, n_min=0, n_max=200, Pe=0.7)
    chosen_model = pure_IOLIN.generate()
    assert chosen_model is not None
