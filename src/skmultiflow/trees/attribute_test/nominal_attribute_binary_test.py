from skmultiflow.trees.attribute_test import InstanceConditionalTest
from skmultiflow.rules.base_predicate import Predicate


class NominalAttributeBinaryTest(InstanceConditionalTest):
    def __init__(self, att_idx, att_value, branch_mapping):
        super().__init__()
        self._att_idx = att_idx
        self._att_value = att_value
        self._branch_mapping = branch_mapping

    def branch_for_instance(self, X):
        if self._att_idx > len(X) or self._att_idx < 0:
            return -1
        else:
            # Access the branch mapping or returns 1 if the attribute value
            # was never seen before
            return self._branch_mapping.get(X[self._att_idx], 1)

    @staticmethod
    def max_branches():
        return 2

    def describe_condition_for_branch(self, branch):
        condition = ' = ' if branch == 0 else ' != '
        return 'Attribute {}{}{}'.format(
            self._att_idx, condition, self._att_value
        )

    def branch_rule(self, branch):
        condition = '==' if branch == 0 else '!='
        return Predicate(self._att_idx, condition, self._att_value)

    def get_atts_test_depends_on(self):
        return [self._att_idx]
