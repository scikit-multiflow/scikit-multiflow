import os
import shutil

from skmultiflow.trees import MultipleModel
from skmultiflow.trees import IfnClassifierMulti

alpha = 0.99
test_tmp_folder = "tmpMultipleModel"


def _setup_test_env():
    if not os.path.isdir(test_tmp_folder):
        os.mkdir(test_tmp_folder)


def _clean_test_env():
    shutil.rmtree(test_tmp_folder, ignore_errors=True)


def test_pure_multiple():
    _setup_test_env()
    ifn = IfnClassifierMulti(alpha)
    multiple_model = MultipleModel(ifn, test_tmp_folder, n_min=0, n_max=200, Pe=0.7)
    chosen_model = multiple_model.generate()
    assert chosen_model is not None
    _clean_test_env()


test_pure_multiple()