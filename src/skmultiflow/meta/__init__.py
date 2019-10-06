"""
The :mod:`skmultiflow.meta` module includes meta learning methods.
"""

from .accuracy_weighted_ensemble import AccuracyWeightedEnsemble
from .adaptive_random_forest_classifier import AdaptiveRandomForestClassifier
from .adaptive_random_forest_regressor import AdaptiveRandomForestRegressor
from .additive_expert_ensemble import AdditiveExpertEnsemble
from .batch_incremental import BatchIncremental
from .classifier_chains import ClassifierChain
from .dynamic_weighted_majority import DynamicWeightedMajority
from .learn_nse import LearnNSE
from .learn_pp import LearnPP
from .leverage_bagging import LeverageBagging
from .classifier_chains import MonteCarloClassifierChain
from .multi_output_learner import MultiOutputLearner
from .online_adac2 import OnlineAdaC2
from .online_boosting import OnlineBoosting
from .online_csb2 import OnlineCSB2
from .online_rus_boost import OnlineRUSBoost
from .online_smote_bagging import OnlineSMOTEBagging
from .online_under_over_bagging import OnlineUnderOverBagging
from .oza_bagging import OzaBagging
from .oza_bagging_adwin import OzaBaggingAdwin
from .classifier_chains import ProbabilisticClassifierChain
from .regressor_chains import RegressorChain


__all__ = ["AccuracyWeightedEnsemble", "AdaptiveRandomForestClassifier",
           "AdaptiveRandomForestRegressor", "AdditiveExpertEnsemble",
           "BatchIncremental", "ClassifierChain", "DynamicWeightedMajority",
           "LearnNSE", "LearnPP", "LeverageBagging",
           "MonteCarloClassifierChain", "MultiOutputLearner", "OnlineAdaC2",
           "OnlineBoosting", "OnlineCSB2", "OnlineRUSBoost",
           "OnlineSMOTEBagging", "OnlineUnderOverBagging",
           "OzaBagging", "OzaBaggingAdwin", "ProbabilisticClassifierChain",
           "RegressorChain"]
