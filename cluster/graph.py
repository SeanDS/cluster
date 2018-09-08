import networkx as nx

class Graph(nx.DiGraph):
    """Directed graph"""
    def get_edge_value(self, first_node, second_node):
        """Get value of edge between specified nodes
        Returns None if no value is defined.
        """
        data = self.get_edge_data(first_node, second_node)

        if data is None:
            raise ValueError("no edge between specified nodes")

        return data.get("value", None)

    def get_node_value(self, node):
        """Gets the value of a node"""
        return self.nodes[node]["value"]

    def set_node_value(self, node, value):
        """Sets the value of a node"""
        self.nodes[node]["value"] = value

    def has_cycle(self, node):
        try:
            nx.algorithms.cycles.find_cycle(self, node)
        except nx.NetworkXNoCycle:
            return False

        return True

    def adjacent_edges(self, node):
        yield from self.in_edges(node)
        yield from self.out_edges(node)
