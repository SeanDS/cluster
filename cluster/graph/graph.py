"""Graph data structures and algorithms.

A graph is typically represented as G=(V,E) where V are vertices and E are
edges. All vertices in a graph have unique ids. Edges are directed edges and are
identified by an ordered pair of vertices (v1, v2). All edges are unique ordered
pairs. Associated with each edge is a value.

A graph is implemented as a dictionary of which the keys are vertices.
Associated with each vertex is (again) a dictionary of which the keys are the
vertices to which there is an edge. Associated with each edge is a value. (A
graph is implemented as a dictionary of dictionaries).

The reverse of the graph is also stored and kept up to date, for fast
determination of incoming edges and other algorithms."""

import logging

LOGGER = logging.getLogger(__name__)

class Graph:
    """A weighted directed graph"""

    def __init__(self, graph=None):
        # forward and reverse edges are stored in a dictionary of dictionaries
        self._forward = {}
        self._reverse = {}

        # copy input graph
        if graph:
            # copy vertices
            list(map(self.add_vertex, graph.vertices()))

            # set up edges between vertices
            list(map(self.set, [(v, w, graph.get(v, w)) for v, w in graph.edges()]))

    def add_vertex(self, vertex):
        """Add vertex to graph

        :param vertex: vertex to add
        """

        if vertex in self._forward:
            # already in dict
            return

        # add vertex to each dict
        self._forward[vertex] = {}
        self._reverse[vertex] = {}

    def remove_vertex(self, vertex):
        """Remove vertex and incident edges

        :param vertex: vertex to remove
        """

        if vertex not in self._forward:
            raise Exception("Vertex not in graph")

        # remove edges going to and from vertex
        list(map(lambda u: self.remove_edge(u, vertex), self.ingoing_vertices(vertex)))
        list(map(lambda w: self.remove_edge(vertex, w), self.outgoing_vertices(vertex)))

        # remove vertex in dicts
        del self._forward[vertex]
        del self._reverse[vertex]

    def add_edge(self, v1, v2, value=1):
        """Add edge with optional value

        :param v1: first vertex
        :param v2: second vertex
        :param value: optional value
        """

        # add vertices
        if v1 not in self._forward:
            self.add_vertex(v1)

        if v2 not in self._forward:
            self.add_vertex(v2)

        # add edge
        if v2 not in self._forward[v1]:
            self._forward[v1][v2] = value

        # and the reverse edge
        if v1 not in self._reverse[v2]:
            self._reverse[v2][v1] = value

    def remove_edge(self, v1, v2):
        """Remove edge

        :param v1: first vertex
        :param v2: second vertex
        """

        if not self.has_edge(v1,v2):
            raise Exception("Edge not in graph")

        # remove edges from dicts
        del self._forward[v1][v2]
        del self._reverse[v2][v1]

    def has_vertex(self, vertex):
        """Check if this graph contains the specified vertex

        :param vertex: vertex to check
        :returns: True if the graph contains the vertex, False otherwise
        :rtype: boolean
        """

        return vertex in self._forward

    def has_edge(self, v1, v2):
        """Check if this graph contains the edge specified by the two vertices

        :param v1: first vertex
        :param v2: second vertex
        :returns: True if the graph contains the edge, False otherwise
        :rtype: boolean
        """

        if v1 not in self._forward:
            return False

        return v2 in self._forward[v1]

    def get(self, v1, v2):
        """Get value of edge

        :param v1: first vertex
        :param v2: second vertex
        :returns: value of edge
        :rtype: hashable
        """

        return self._forward[v1][v2]

    def set(self, v1, v2, value):
        """Set value of edge, adding it if it doesn't yet exist

        :param v1: first vertex
        :param v2: second vertex
        :param value: vertex value
        """

        if not self.has_edge(v1, v2):
            # add new edge and set value
            self.add_edge(v1, v2, value)
        else:
            # set edge value
            self._forward[v1][v2] = value
            self._reverse[v2][v1] = value

    def set_bi_edge(self, v1, v2, value):
        """Set value of bidirectional edge, adding them if they doesn't yet \
        exist

        :param v1: first vertex
        :param v2: second vertex
        :param value: vertex value
        """

        self.set(v1, v2, value)
        self.set(v2, v1, value)

    def vertices(self):
        """Get a list of vertices in this graph"""

        return list(self._forward.keys())

    def edges(self):
        """Get a list of the edges in this graph"""

        # empty list
        l = []

        for i in self._forward:
            for j in self._forward[i]:
                l.append((i, j))

        return l

    def outgoing_vertices(self, vertex):
        """Get a list of vertices connected from the specified vertex via an \
        edge

        :param vertex: vertex to use as a reference
        """

        # look up forward graph
        return list(self._forward[vertex].keys())

    def ingoing_vertices(self, vertex):
        """Get a list of vertices connected to the specified vertex via an \
        edge

        :param vertex: vertex to use as a reference
        """

        # look up the reverse graph
        return list(self._reverse[vertex].keys())

    def adjacent_vertices(self, vertex):
        """Get a list of adjacent (ingoing or outgoing) vertices

        :param vertex: vertex to use as a reference
        """

        # return union of ingoing and outgoing vertices
        return list(set(self.ingoing_vertices(vertex)).union(set(self.outgoing_vertices(vertex))))

    def ingoing_edges(self, vertex):
        """Get a list of edges connecting towards the specified vertex

        :param vertex: vertex to retrieve edges for
        """

        return [(v, vertex) for v in self.ingoing_vertices(vertex)]

    def outgoing_edges(self, vertex):
        """Get a list of edges connecting away from the specified vertex

        :param vertex: vertex to retrieve edges for
        """

        return [(v, vertex) for v in self.outgoing_vertices(vertex)]

    def adjacent_edges(self, vertex):
        """Get a list of ingoing and outgoing edges

        :param vertex: vertex to retrieve edges for
        """

        return self.ingoing_edges(vertex) + self.outgoing_edges(vertex)

    def path(self, start, end):
        """Gets an arbitrary path (list of vertices) from start to end

        If start is equal to end, then the path is a cycle
        If there is no path between start and end, then an empty list is
        returned

        :param start: start vertex
        :param end: end vertex
        :returns: list of vertices connecting start and end
        :rtype: list
        """

        # map from vertices to shortest path to that key vertex
        trails = {}

        # set start vertex
        trails[start] = [start]

        # list of vertices to consider
        consider = [start]

        # loop until there are no vertices to consider
        while len(consider) > 0:
            # next key to consider
            key = consider.pop()

            # current path
            this_path = trails[key]

            for v in self.outgoing_vertices(key):
                if v == end:
                    # found the end, so return the path we have taken
                    return this_path + [v]
                elif v not in trails:
                    # add vertex to trails taken
                    trails[v] = this_path + [v]

                    # add the vertex to the list to be searched
                    consider.append(v)
                elif len(this_path) + 1 < len(trails[v]):
                    # this trail is shorter than a previous one, so overwrite
                    trails[v] = this_path + [v]

        # no path found
        return []

    def connected(self, vertex, ingoing=True, outgoing=True):
        """Gets vertices x connected to the specified vertex by following \
        edges connected to specified vertex or x

        The edges can be unidirectional or bidirectional as specified by the
        optional ingoing and outgoing flags.

        Specified vertex is not included in the result.

        :param vertex: vertex to start at
        :param ingoing: include ingoing edges
        :param outgoing: include outgoing edges
        """

        # work out which edge function to use
        if ingoing and outgoing:
            vertex_edge_func = self.adjacent_vertices
        elif ingoing:
            vertex_edge_func = self.ingoing_vertices
        else:
            vertex_edge_func = self.outgoing_vertices

        # vertices being searched
        front = [vertex]

        # connected vertices
        connected = {}

        # loop until there are no more connected vertices to follow
        while len(front) > 0:
            # get next connected vertex
            x = front.pop()

            # add vertex to dict
            if x not in connected:
                # add vertex
                connected[x] = 1

                # add connected vertices to search list
                front += vertex_edge_func(x)

        # delete the supplied vertex
        del connected[vertex]

        # convert result to a list
        return list(connected)

    def __str__(self):
        s = ""

        for i in self._forward:
            v = ", ".join(["{0}: {1}".format(str(j), str(self.get(i, j))) for j in self._forward[i]])

            s += "{0}: {{1}}".format(str(i), v)

        return "{{0}}".format(s)
