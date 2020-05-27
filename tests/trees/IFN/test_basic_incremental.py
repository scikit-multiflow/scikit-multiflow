from skmultiflow.trees import BasicIncremental
from skmultiflow.trees import IfnClassifier

alpha = 0.99


def test_basic_incremental(tmpdir):
    dir = tmpdir.mkdir("tmpBasicIncremental")
    ifn = IfnClassifier(alpha)
    basic_incremental = BasicIncremental(ifn, dir, n_min=0, n_max=300, Pe=0.7)
    last_model = basic_incremental.generate()
    assert last_model is not None
