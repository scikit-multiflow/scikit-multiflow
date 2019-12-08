"""
The :mod:`skmultiflow.meta` module includes meta learning methods.
"""

from .adaptive_random_forests import AdaptiveRandomForestClassifier
from .batch_incremental import BatchIncrementalClassifier
from .leverage_bagging import LeverageBagging
from .oza_bagging import OzaBagging
from .oza_bagging_adwin import OzaBaggingAdwin
from .classifier_chains import ClassifierChain
from .classifier_chains import ProbabilisticClassifierChain
from .classifier_chains import MonteCarloClassifierChain
from .regressor_chains import RegressorChain
from .multi_output_learner import MultiOutputLearner
from .learn_pp import LearnPP
from .learn_nse import LearnPPNSEClassifier
from .accuracy_weighted_ensemble import AccuracyWeightedEnsembleClassifier
from .dynamic_weighted_majority import DynamicWeightedMajorityClassifier
from .additive_expert_ensemble import AdditiveExpertEnsembleClassifier
from .online_boosting import OnlineBoosting
from .online_adac2 import OnlineAdaC2
from .online_csb2 import OnlineCSB2
from .online_under_over_bagging import OnlineUnderOverBagging
from .online_rus_boost import OnlineRUSBoost
from .online_smote_bagging import OnlineSMOTEBagging
from .batch_incremental import BatchIncremental   # remove in v0.7.0
from .accuracy_weighted_ensemble import AccuracyWeightedEnsemble   # remove in v0.7.0
from .adaptive_random_forests import AdaptiveRandomForest   # remove in v0.7.0
from .additive_expert_ensemble import AdditiveExpertEnsemble   # remove in v0.7.0
from .dynamic_weighted_majority import DynamicWeightedMajority   # remove in v0.7.0
from .learn_nse import LearnNSE   # remove in v0.7.0


__all__ = ["AdaptiveRandomForestClassifier", "BatchIncrementalClassifier", "LeverageBagging", "OzaBagging",
           "OzaBaggingAdwin", "ClassifierChain", "ProbabilisticClassifierChain", "MonteCarloClassifierChain",
           "RegressorChain", "MultiOutputLearner", "LearnPP", "LearnPPNSEClassifier",
           "AccuracyWeightedEnsembleClassifier", "DynamicWeightedMajorityClassifier",
           "AdditiveExpertEnsembleClassifier", "OnlineSMOTEBagging", "OnlineRUSBoost", "OnlineCSB2", "OnlineAdaC2",
           "OnlineUnderOverBagging", "OnlineBoosting", "BatchIncremental", "AccuracyWeightedEnsemble",
           "AdaptiveRandomForest", "AdditiveExpertEnsemble", "DynamicWeightedMajority", "LearnNSE"]
