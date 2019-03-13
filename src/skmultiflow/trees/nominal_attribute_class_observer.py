from skmultiflow.trees.attribute_class_observer import AttributeClassObserver
from skmultiflow.trees.attribute_split_suggestion import AttributeSplitSuggestion
from skmultiflow.trees.nominal_attribute_binary_test import NominalAttributeBinaryTest
from skmultiflow.trees.nominal_attribute_multiway_test import NominalAttributeMultiwayTest


class NominalAttributeClassObserver(AttributeClassObserver):
    """ Class for observing the class data distribution for a nominal attribute.
    This observer monitors the class distribution of a given attribute.
    Used in naive Bayes and decision trees to monitor data statistics on leaves.

    """

    def __init__(self):
        super().__init__()
        self._total_weight_observed = 0.0
        self._missing_weight_observed = 0.0
        self._att_val_dist_per_class = {}

    def observe_attribute_class(self, att_val, class_val, weight):
        if att_val is None:
            self._missing_weight_observed += weight
        else:
            try:
                val_dist = self._att_val_dist_per_class[class_val]
            except KeyError:
                self._att_val_dist_per_class[class_val] = {att_val: 0.0}
                self._att_val_dist_per_class = dict(sorted(self._att_val_dist_per_class.items()))
            try:
                self._att_val_dist_per_class[class_val][att_val] += weight
            except KeyError:
                self._att_val_dist_per_class[class_val][att_val] = weight
                self._att_val_dist_per_class[class_val] = dict(sorted(self._att_val_dist_per_class[class_val].items()))

        self._total_weight_observed += weight

    def probability_of_attribute_value_given_class(self, att_val, class_val):
        obs = self._att_val_dist_per_class.get(class_val, None)
        if obs is not None:
            value = obs[att_val] if att_val in obs else 0.0
            return (value + 1.0) / (sum(obs.values()) + len(obs))
        return 0.0

    def get_best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):
        best_suggestion = None
        if not binary_only:
            post_split_dist = self.get_class_dist_from_multiway_split()
            merit = criterion.get_merit_of_split(pre_split_dist, post_split_dist)
            best_suggestion = AttributeSplitSuggestion(NominalAttributeMultiwayTest(att_idx),
                                                       post_split_dist, merit)
        att_values = set([att_val for class_val in self._att_val_dist_per_class.values() for att_val in class_val])
        for att_val in att_values:
            post_split_dist = self.get_class_dist_from_binary_split(att_val)
            merit = criterion.get_merit_of_split(pre_split_dist, post_split_dist)
            if best_suggestion is None or merit > best_suggestion.merit:
                best_suggestion = AttributeSplitSuggestion(NominalAttributeBinaryTest(att_idx, att_val),
                                                           post_split_dist, merit)
        return best_suggestion

    def get_class_dist_from_multiway_split(self):
        resulting_dist = {}
        for i, att_val_dist in self._att_val_dist_per_class.items():
            for j, value in att_val_dist.items():
                if j not in resulting_dist:
                    resulting_dist[j] = {}
                if i not in resulting_dist[j]:
                    resulting_dist[j][i] = 0.0
                resulting_dist[j][i] += value

        distributions = [dict(sorted(value.items())) for value in resulting_dist.values()]
        return distributions

    def get_class_dist_from_binary_split(self, val_idx):
        equal_dist = {}
        not_equal_dist = {}
        for i, att_val_dist in self._att_val_dist_per_class.items():
            for j, value in att_val_dist.items():
                if j == val_idx:
                    if i not in equal_dist:
                        equal_dist[i] = 0.0
                    equal_dist[i] += value
                else:
                    if i not in not_equal_dist:
                        not_equal_dist[i] = 0.0
                    not_equal_dist[i] += value
        return [equal_dist, not_equal_dist]
