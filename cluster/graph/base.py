from networkx import DiGraph

class Graph(DiGraph):
    """Directed graph"""
    def value(self, first_node, second_node):
        """Get value of edge between specified nodes

        Returns None if no value is defined.
        """
        data = self.get_edge_data(first_node, second_node)

        if data is None:
            raise ValueError("no edge between specified nodes")

        return data.get("value", None)
