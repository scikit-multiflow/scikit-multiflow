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
        density = [[[[0] * 4 for _ in range(evaluator.n_intervals)] for _ in range(evaluator.stream.n_features)]
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
                    TN1 = evaluator.distribution_table[chunk1][feature][interval][TN]
                    TN0 = evaluator.distribution_table[chunk0][feature][interval][TN]

                    FP1 = evaluator.distribution_table[chunk1][feature][interval][FP]
                    FP0 = evaluator.distribution_table[chunk0][feature][interval][FP]

                    FN1 = evaluator.distribution_table[chunk1][feature][interval][FN]
                    FN0 = evaluator.distribution_table[chunk0][feature][interval][FN]

                    TP1 = evaluator.distribution_table[chunk1][feature][interval][TP]
                    TP0 = evaluator.distribution_table[chunk0][feature][interval][TP]

                    density_TP, density_FN, density_TN, density_FP = 0, 0, 0, 0
                    if TN0 > 0:
                        density_TN = (TN1 - TN0) / TN0
                    if TP0 > 0:
                        density_TP = (TP1 - TP0) / TP0
                    if FN0 > 0:
                        density_FN = (FN1 - FN0) / FN0
                    if FP0 > 0:
                        density_FP = (FP1 - FP0) / FP0
                    densities = [density_TN, density_FP, density_FN, density_TP]

                    for i in range(4):
                        density[chunk0][feature][interval][i] = densities[i]
                    # density differences is density1 in density0
                    #density_difference = density_1 - density_0
                    #density[chunk0][feature][interval][1] = density_difference

                for interval in range(evaluator.n_intervals):
                    subset_TN[chunk0][feature].extend(
                        [density[chunk0][feature][interval][TN]] * t0[TN][interval])
                    subset_FP[chunk0][feature].extend(
                        [density[chunk0][feature][interval][FP]] * t0[FP][interval])
                    subset_FN[chunk0][feature].extend(
                        [density[chunk0][feature][interval][FN]] * t0[FN][interval])
                    subset_TP[chunk0][feature].extend(
                        [density[chunk0][feature][interval][TP]] * t0[TP][interval])

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
