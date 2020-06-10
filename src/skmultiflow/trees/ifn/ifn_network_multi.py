from skmultiflow.trees.ifn.ifn_network import Node, IfnNetwork


# target node. index = 0/1
class ClassNode(Node):
    def __init__(self, index, target=None):
        super().__init__(index)
        self.target_number = target


class IfnNetworkMulti(IfnNetwork):
    def __init__(self):
        super().__init__()
        self.target_layer = {}

    def build_target_layer(self, num_of_classes, target_number):
        if len(num_of_classes) != 0:
            self.target_layer[target_number] = []
            for i in num_of_classes:
                self.target_layer[target_number].append(ClassNode(i))
