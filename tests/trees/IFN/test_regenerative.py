import os
import shutil
from skmultiflow.trees import OnlineNetworkRegenerative
from src.skmultiflow.trees import IfnClassifierMulti

alpha = 0.99
test_tmp_folder = "tmpOLIN"


def _setup_test_env():
    if not os.path.isdir(test_tmp_folder):
        os.mkdir(test_tmp_folder)


def _clean_test_env():
    shutil.rmtree(test_tmp_folder, ignore_errors=True)


def test_OLIN():
    _setup_test_env()
    ifn = IfnClassifierMulti(alpha)
    regenerative = OnlineNetworkRegenerative(ifn, test_tmp_folder, n_min=0, n_max=60, Pe=0.7)
    last_model = regenerative.generate()
    assert last_model is not None
    _clean_test_env()



test_OLIN()