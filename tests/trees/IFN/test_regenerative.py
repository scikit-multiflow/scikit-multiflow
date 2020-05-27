from skmultiflow.trees import OnlineNetworkRegenerative
from src.skmultiflow.trees import IfnClassifier

alpha = 0.99


def test_regenerative(tmpdir):
    dir = tmpdir.mkdir("tmpOLIN")
    ifn = IfnClassifier(alpha)
    regenerative = OnlineNetworkRegenerative(ifn, dir, n_min=0, n_max=1000, Pe=0.7)
    regenerative.generate()