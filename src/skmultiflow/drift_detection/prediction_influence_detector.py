from skmultiflow.evaluation.evaluate_influential import EvaluateInfluential
from scipy.stats import ranksums
from skmultiflow.core import BaseSKMObject
from abc import ABCMeta, abstractmethod


class predictionInfluenceDetector(BaseSKMObject, metaclass=ABCMeta):

    def __init__(self, run=None, strategy=None):
        self.table_influence_on_positive = []
        self.table_influence_on_negative = []
        self.run = run
        self.strategy = strategy

    def calculate_density(self, evaluator):
        # table = tn, fp, fn, t
        # create list that has two values (difference in density in time0 and time1 of TP and FN,
        #                               and difference in density in time0 and time1 of TN and FP per interval
        TN, FP, FN, TP = 0, 1, 2, 3
        density = [[[[0] * 2 for _ in range(evaluator.n_intervals)] for _ in range(evaluator.stream.n_features)]
                   for _ in range(evaluator.n_time_windows - 1)]
        subset_TP = [[[] for _ in range(evaluator.stream.n_features)] for _ in range(evaluator.n_time_windows - 1)]
        subset_FN = [[[] for _ in range(evaluator.stream.n_features)] for _ in range(evaluator.n_time_windows - 1)]
        subset_TN = [[[] for _ in range(evaluator.stream.n_features)] for _ in range(evaluator.n_time_windows - 1)]
        subset_FP = [[[] for _ in range(evaluator.stream.n_features)] for _ in range(evaluator.n_time_windows - 1)]
        for chunk0 in range(evaluator.n_time_windows - 1):
            chunk1 = chunk0 + 1
            for feature in range(evaluator.stream.n_features):
                # dist table is organized like this: [tn, fp, fn, tp],[tn, fp, fn, tp],[tn, fp, fn, tp],[tn, fp, fn, tp]
                # next step will create the following: [tn, tn, tn, tn],[fp,fp,fp,fp] etc (if there are 4 intervals)
                # print("distribution table: ", self.distribution_table)

                t0 = list(zip(*evaluator.distribution_table[chunk0][feature]))
                # create table with density differences of TP and FN (0), and TN and FP (1), per interval
                for interval in range(evaluator.n_intervals):
                    # calculate density of positive instances = instances of TP + FN / window size
                    density_0 = (evaluator.distribution_table[chunk0][feature][interval][TP] +
                                 evaluator.distribution_table[chunk0][feature][interval][FN]) / evaluator.window_size
                    density_1 = (evaluator.distribution_table[chunk1][feature][interval][TP] +
                                 evaluator.distribution_table[chunk1][feature][interval][FN]) / evaluator.window_size
                    # density differences is density1 - density0
                    density_difference = density_1 - density_0

                    # fill in density difference in list
                    density[chunk0][feature][interval][0] = density_difference

                    # calculate density of negative instances
                    density_0 = (evaluator.distribution_table[chunk0][feature][interval][TN] +
                                 evaluator.distribution_table[chunk0][feature][interval][FP]) / evaluator.window_size
                    density_1 = (evaluator.distribution_table[chunk1][feature][interval][TN] +
                                 evaluator.distribution_table[chunk1][feature][interval][FP]) / evaluator.window_size
                    # density differences is density1 in density0
                    density_difference = density_1 - density_0
                    density[chunk0][feature][interval][1] = density_difference

                for interval in range(evaluator.n_intervals):
                    # add the amount of instances per interval that is belonging to subset
                    # so if feature0, interval 0 has 6 TP instances, and the calculated difference
                    # in density of feature0,
                    # interval 0 is 0.07, you will extend the subset with [0.07,0.07,0.07,0.07,0.07,0.07]
                    subset_TP[chunk0][feature].extend(
                        [density[chunk0][feature][interval][0]] * t0[TP][interval])
                    subset_FN[chunk0][feature].extend(
                        [density[chunk0][feature][interval][0]] * t0[FN][interval])
                    subset_FP[chunk0][feature].extend(
                        [density[chunk0][feature][interval][1]] * t0[FP][interval])
                    subset_TN[chunk0][feature].extend(
                        [density[chunk0][feature][interval][1]] * t0[TN][interval])
        predictionInfluenceDetector.test_density(self, evaluator, subset_TN, subset_FP, subset_FN, subset_TP)

    def test_density(self, evaluator, subset_TN, subset_FP, subset_FN, subset_TP):
        for chunk in range(evaluator.n_time_windows - 1):
            for feature in range(evaluator.stream.n_features):
                TP = subset_TP[chunk][feature]
                FN = subset_FN[chunk][feature]
                TN = subset_TN[chunk][feature]
                FP = subset_FP[chunk][feature]
                # positive instances:
                mean_subset_TP = 0
                mean_subset_FN = 0
                if len(subset_TP[chunk][feature]) > 0:
                    mean_subset_TP = sum(TP) / len(TP)
                if len(subset_FN[chunk][feature]) > 0:
                    mean_subset_FN = sum(FN) / len(FN)
                if len(TP) > 10 and len(FN) > 10:
                    test = ranksums(TP, FN)
                    result = test.pvalue
                    self.table_influence_on_positive.append([self.run, self.strategy, feature, chunk, len(TP),
                                                             mean_subset_TP, len(FN),
                                                             mean_subset_FN,
                                                             abs(mean_subset_TP - mean_subset_FN),
                                                             result])

                # negative instances
                mean_subset_TN = 0
                mean_subset_FP = 0
                if len(TN) > 0:
                    mean_subset_TN = sum(TN) / len(TN)
                if len(FP) > 0:
                    mean_subset_FP = sum(FP) / len(FP)
                if len(TN) > 10 and len(FP) > 10:
                    test = ranksums(TN, FP)
                    result = test.pvalue
                    self.table_influence_on_negative.append([self.run, self.strategy, feature, chunk, len(TN),
                                                             mean_subset_TN, len(FP),
                                                             mean_subset_FP,
                                                             abs(mean_subset_TN - mean_subset_FP),
                                                             result])
