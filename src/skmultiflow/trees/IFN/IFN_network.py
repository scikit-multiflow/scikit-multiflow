# index of the layer
class IfnNode:
    def __init__(self, index):
        self.index = index


class IfnRootNode(IfnNode):
    def __init__(self):
        super().__init__(0)
        self.first_layer = None

    def set_layer(self, layer):
        self.first_layer = layer


# target node. index = 0/1
class IfnClassNode(IfnNode):
    def __init__(self, index):
        super().__init__(index)


class IfnAttributeNode(IfnNode):
    def __init__(self, index, attribute_value, prev_node, layer, partial_x, partial_y, is_terminal=False):
        """

        :param index: of the node
        :param inner_index: of the node inside the layer
        :param prev_node: previous node connected to this
        :param layer:
        :param partial_x: samples arrives to the node
        :param partial_y: classes attributes of partial_x
        :param is_terminal: false if the node isn't terminal
        """
        super().__init__(index)
        self.attribute_value = attribute_value
        self.prev_node = prev_node
        self.layer = layer
        self.is_terminal = is_terminal
        self.weight_probability_pair = {}
        self.partial_x = partial_x
        self.partial_y = partial_y

    def set_terminal(self):
        self.is_terminal = True

    def set_weight_probability_pair(self, weight_probability_pair):
        if self.is_terminal:
            self.weight_probability_pair = weight_probability_pair


# layer contains attribute nodes
class IfnHiddenLayer:
    def __init__(self, index):
        self.index = index
        self.next_layer = None
        self.nodes = None
        self.is_continuous = False
        self.split_points = []

    def set_nodes(self, nodes):
        self.nodes = nodes

    def get_node(self, index):
        for node in self.nodes:
            if node.index == index:
                return node
        return None

    def get_nodes(self):
        return self.nodes


class IfnNetwork:
    def __init__(self):
        self.target_layer = []
        self.root_node = IfnRootNode()

    def build_target_layer(self, num_of_classes):
        if len(num_of_classes) != 0:
            for i in num_of_classes:
                self.target_layer.append(IfnClassNode(i))

    def create_network_structure_file(self, path):
        f = open(path, "w+")
        f.write("Network Structure:" + "\n\n")

        curr_layer = self.root_node.first_layer
        first_line = "0"
        for node in curr_layer.nodes:
            first_line = first_line + " " + str(node.index)
        f.write(first_line + "\n")
        while curr_layer.next_layer is not None:
            for curr_node in curr_layer.nodes:
                curr_line = str(curr_node.index)
                for next_node in curr_layer.next_layer.nodes:
                    # check if there is link between next node to curr node
                    if next_node.prev_node == curr_node.index:
                        curr_line = curr_line + " " + str(next_node.index)
                if ' ' in curr_line:
                    f.write(curr_line + "\n")
            curr_layer = curr_layer.next_layer
        f.close()
