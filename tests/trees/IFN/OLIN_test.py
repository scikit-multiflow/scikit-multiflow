import os
import shutil
from skml import OnlineNetwork
from skml import IfnClassifier


alpha = 0.99
test_tmp_folder = "tmpOLIN"


def _setup_test_env():
    if not os.path.isdir(test_tmp_folder):
        os.mkdir(test_tmp_folder)


def _clean_test_env():
    shutil.rmtree(test_tmp_folder, ignore_errors=True)


def test_OLIN():
    _setup_test_env()
    ifn = IfnClassifier(alpha)
    OLIN = OnlineNetwork(ifn, test_tmp_folder,n_min=0, n_max=60, Pe=0.7)
    OLIN.regenerate()
    _clean_test_env()



test_OLIN()