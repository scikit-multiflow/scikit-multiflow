from skmultiflow.trees.attribute_observer import AttributeClassObserver
from skmultiflow.trees.attribute_test import NominalAttributeBinaryTest
from skmultiflow.trees.attribute_test import NominalAttributeMultiwayTest
from skmultiflow.trees.attribute_split_suggestion import AttributeSplitSuggestion

from collections import Counter


class NominalAttributeRegressionObserver(AttributeClassObserver):
    """ Class for observing the data distribution for a nominal attribute for
    regression.
    """

    def __init__(self):
        super().__init__()
        self._statistics = {}

    def observe_attribute_class(self, att_val, target, weight=1.0):
        if att_val is None or weight is None:
            return
        else:
            if att_val in self._statistics:
                new = Counter({
                    0: weight,
                    1: weight * target,
                    2: weight * target * target
                })
                self._statistics[att_val] += new
            else:
                self._statistics[att_val] = Counter({
                    0: weight,
                    1: weight * target,
                    2: weight * target * target
                })

    def probability_of_attribute_value_given_class(self, att_val, target):
        return 0.0

    def get_best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):
        current_best = None
        if not binary_only:
            post_split_dist = [
                dict(stat) for stat in self._statistics.values()
            ]
            merit = criterion.get_merit_of_split(
                pre_split_dist, post_split_dist
            )
            branch_mapping = {attr_val: branch_id for branch_id, attr_val in
                              enumerate(self._statistics)}
            current_best = AttributeSplitSuggestion(
                NominalAttributeMultiwayTest(att_idx, branch_mapping),
                post_split_dist, merit
            )

        pre_split_counter = Counter(pre_split_dist)
        for att_val in self._statistics:
            actual_dist = self._statistics[att_val]
            remaining_dist = dict(pre_split_counter - actual_dist)
            post_split_dist = [dict(actual_dist), remaining_dist]

            merit = criterion.get_merit_of_split(pre_split_dist,
                                                 post_split_dist)

            if current_best is None or merit > current_best.merit:
                nom_att_binary_test = NominalAttributeBinaryTest(att_idx,
                                                                 att_val)
                current_best = AttributeSplitSuggestion(
                    nom_att_binary_test, post_split_dist, merit
                )

        return current_best
