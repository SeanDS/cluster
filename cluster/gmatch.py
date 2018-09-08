# graph matching algorithm(s)

from .graph import Graph

def gmatch(pattern, reference):
    """Match pattern graph to reference graph.

       Pattern matches are subgraphs of reference (subgraph isomorphisms).
       (a subgraph is a subset of variables and subset of edges)
       Any vertices in the pattern that are equal to some vertex in reference, they are matched exactly.
       Otherwise, vertices in pattern are considered variables.
       Returns a list of solutions.
       Each solution is a Map from pattern vertices to reference vertices (and vice versa).
    """

    if not isinstance(pattern, FanGraph):
        pattern = FanGraph(pattern)
    if not isinstance(reference, FanGraph):
        reference = FanGraph(reference)


    # For each pattern vertex:
    #  match with all vertices in reference that have at least same fanin and fanout.
    #  also match if pattern vertex in reference (same object or equal)
    #  Then combine matches (patvar, refvar) with existing partial solutions if:
    #    refvar still free in partial solution
    #    all edges adjacent to pattern vertex are also in reference graph

    solutions = None
    for patvar in pattern.nodes:
        if reference.has_node(patvar):
            matches = [patvar]
        else:
            fanin = pattern.fanin(patvar)
            fanout = pattern.fanout(patvar)
            inumbers = [n for n in reference.fanin_numbers() if n>=fanin]
            onumbers = [n for n in reference.fanout_numbers() if n>=fanout]
            inmatches = []
            for n in inumbers:
                inmatches += reference.infan(n)
            outmatches = []
            for n in onumbers:
                outmatches += reference.outfan(n)
            matches = set(inmatches).intersection(outmatches)

        newsolutions = []

        if solutions == None:
            for refvar in matches:
                s = {patvar:refvar, refvar:patvar}
                newsolutions.append(s)
        else:
            for refvar in matches:
                for olds in solutions:
                    news = dict(olds)
                    news[patvar] = refvar
                    news[refvar] = patvar
                    consistent = True
                    # check for no double assignments
                    if patvar in olds:
                        if olds[patvar] != refvar:
                            consistent = False
                    if refvar in olds:
                        if olds[refvar] != patvar:
                            consistent = False
                    # check edges
                    for pe in pattern.adjacent_edges(patvar):
                        (pv1,pv2) = pe
                        if pv1 not in news or pv2 not in news:
                            continue
                        rv1 = news[pv1]
                        rv2 = news[pv2]
                        if not reference.has_edge(rv1,rv2):
                            consistent = False
                            break
                    if consistent:
                        newsolutions.append(news)
                #for
            #for
        #if
        solutions = newsolutions
    #for
    return solutions
#gmatch


class FanGraph(Graph):
    """A graph with updated fan-in and fan-out numbers"""

    def __init__(self, graph=None):
        super().__init__()

        self._dict = {}
        """the edges are stored in a dictionary of dictionaries"""
        self._reverse = {}
        """the reverse graph is stored here"""

        self._fanin = {}
        """map from vertices to fan-in number"""
        self._fanout = {}
        """map from vertices to fan-out number"""
        self._infan = {}
        """map from fan-in numbers to vertices with that fan-in"""
        self._outfan = {}
        """map from fan-out numbers to vertices with that fan-out"""
        # copy input graph
        if graph:
            for v in graph.nodes:
                self.add_node(v)
            for e in graph.edges:
                (v, w) = e
                self.add_edge(v, w, value=graph.get_edge_value(v, w))
    #end __init__

    def add_node(self, v):
        super().add_node(v)

        self._set_fanin(v, 0)
        self._set_fanout(v, 0)

    def add_edge(self, v1, v2, **kwargs):
        "Add edge from v1 to v2 with optional value."
        super().add_edge(v1, v2, **kwargs)

        # increment fan-out for v1
        self._set_fanout(v1, self._fanout[v1] + 1)
        # increment fan-in for v2
        self._set_fanin(v2, self._fanin[v2] + 1)

    def remove_node(self, v):
        super().remove_node(v)

        # remove entries from fan-in and fan-out tables
        self._set_fanin(v, None)
        self._set_fanout(v, None)

    def remove_edge(self, v1, v2):
        super().remove_edge(v1, v2)

        # decrement fan-out for v1
        self._set_fanout(v1, self._fanout[v1] - 1)
        # decrement fan-in for v2
        self._set_fanin(v2, self._fanin[v2] - 1)

    def fanin(self, v):
        """return fan-in number (number of in-going edges)"""
        return self._fanin[v]

    def fanout(self, v):
        """return fan-out number (number of out-going edges)"""
        return self._fanout[v]

    def infan(self, number):
        """return a list of vertices with given fan-in number"""
        if number in self._infan:
            return self._infan[number]
        else:
            return []

    def outfan(self, number):
        """return a list of vertices with given fan-out number"""
        if number in self._outfan:
            return list(self._outfan[number])
        else:
            return []

    def fanin_numbers(self):
        """the set of fan-in numbers,
           i.e. the union of the fan-in numbers of all veretices
        """
        return self._infan

    def fanout_numbers(self):
        """the set of fan-out numbers,
        i.e. the union of the fan-out numbers of all veretices
        """
        return self._outfan

    def roots(self):
        """return a list of vertices with zero fan-in"""
        return self.infan(0)

    def leafs(self):
        """return a list of vertices with zero fan-out"""
        return self.outfan(0)

    def singular(self, number):
        """return a list of vertices with no fan-in and no fan-out"""
        l = self._infan[0]
        for v in l:
            if self._fanout[v] == 0:
                l.append(v)
        return l

    # non-public methods

    def _set_fanin(self, vertex, number):
        # remove previous entry in infan dict
        if vertex in self._fanin: # test for new vertices
            old = self._fanin[vertex]
            l = self._infan[old]
            l.remove(vertex)

            if len(l) == 0:
                del self._infan[old]

        if number == None: # test for deleted vertices
            del self._fanin[vertex]
        else:
            # store new fanin
            self._fanin[vertex] = number
            # store in infan
            if number not in self._infan:
                self._infan[number] = []
            self._infan[number].append(vertex)

    def _set_fanout(self, vertex, number):
        # remove previous entry in outfan dict
        if vertex in self._fanout: # test for new vertices
            old = self._fanout[vertex]
            l = self._outfan[self._fanout[vertex]]
            l.remove(vertex)

            if len(l) == 0:
                del self._outfan[old]

        if number == None: # test for deleted vertices
            del self._fanout[vertex]
        else:
            # store new fanin
            self._fanout[vertex] = number
            # store in infan
            if number not in self._outfan:
                self._outfan[number] = []
            self._outfan[number].append(vertex)


def test():
    print("matching a triangle")
    pattern = Graph()
    pattern.add_edge('x','a')
    pattern.add_edge('x','b')
    pattern.add_edge('y','a')
    pattern.add_edge('y','c')
    pattern.add_edge('z','b')
    pattern.add_edge('z','c')
    pattern.add_edge("distance","x")
    pattern.add_edge("distance","y")
    pattern.add_edge("distance","z")

    reference = Graph()
    reference.add_edge(1,'A')
    reference.add_edge(1,'B')
    reference.add_edge(2,'A')
    reference.add_edge(2,'C')
    reference.add_edge(3,'B')
    reference.add_edge(3,'C')
    reference.add_edge("rigid", 1)
    reference.add_edge("distance", 1)
    reference.add_edge("rigid", 2)
    reference.add_edge("distance", 2)
    reference.add_edge("rigid", 3)
    reference.add_edge("distance", 3)

    s = gmatch(pattern, reference)
    print(s)
    print(len(s),"solutions")


    print("mathing random pattern in random graph")
    pattern = random_graph(3,6,False,"v")
    reference = random_graph(100,200,False,"t")
    s = gmatch(pattern, reference)
    print(s)
    print(len(s),"solutions")


if __name__ == "__main__": test()



