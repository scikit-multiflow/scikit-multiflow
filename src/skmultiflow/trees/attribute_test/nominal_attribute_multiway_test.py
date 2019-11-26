from skmultiflow.trees.attribute_test import InstanceConditionalTest
from skmultiflow.rules.base_predicate import Predicate


class NominalAttributeMultiwayTest(InstanceConditionalTest):
    def __init__(self, att_idx, branch_mapping):
        super().__init__()
        self._att_idx = att_idx
        self._branch_mapping = branch_mapping
        self._reverse_branch_mapping = {
            b: v for v, b in branch_mapping.items()
        }

    def branch_for_instance(self, X):
        if self._att_idx > len(X) or self._att_idx < 0:
            return -1
        else:
            # Return branch for feature value or -1 in case the element was not
            # observed yet
            return self._branch_mapping.get(X[self._att_idx], -1)

    @staticmethod
    def max_branches():
        return -1

    def describe_condition_for_branch(self, branch):
        return 'Attribute {} = {}'.format(
            self._att_idx, self._reverse_branch_mapping[branch]
        )

    def branch_rule(self, branch):
        return Predicate(
            self._att_idx, '==', self._reverse_branch_mapping[branch]
        )

    def get_atts_test_depends_on(self):
        return [self._att_idx]
