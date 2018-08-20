import logging

LOGGER = logging.getLogger(__name__)

class Decomposition:
    """Represents the result of solving a GeometricProblem. A cluster is a list of
       point variable names and a list of solutions for
       those variables. A solution is a dictionary mapping variable names to
       points. The cluster also keeps a list of sub-clusters (GeometricCluster)
       and a set of flags, indicating incidental/structural
       under/overconstrained

       instance attributes:
            variables       - a list of point variable names
            solutions       - a list of solutions. Each solution is a dictionary
                              mapping variable names to :class:`Vector`
                              objects.
            subs            - a list of sub-clusters
            flag            - value                 meaning
                              OK                    well constrained
                              I_OVER                incicental over-constrained
                              I_UNDER               incidental under-constrained
                              S_OVER                structural overconstrained
                              S_UNDER               structural underconstrained
                              UNSOLVED              unsolved
       """

    OK = "well constrained"
    I_OVER = "incidental over-constrained"
    I_UNDER = "incidental under-constrained"
    S_OVER = "structral over-constrained"
    S_UNDER = "structural under-constrained"
    UNSOLVED = "unsolved"

    def __init__(self):
        """initialise an empty new cluster"""

        self.variables = []
        self.solutions = []
        self.subs = []
        self.flag = self.OK

    def __str__(self):
        return self._str_recursive()

    def _str_recursive(result, depth=0, done=None):
        # create indent
        spaces = ""

        for i in range(depth):
            spaces = spaces + "|"

        # make done
        if done is None:
            done = set()

        # recurse
        s = ""

        if result not in done:
            # this one is done...
            done.add(result)

            # recurse
            for sub in result.subs:
                s = s + sub._str_recursive(depth+1, done)

        elif len(result.subs) > 0:
            s = s + spaces + "|...\n"

        # variables
        variables = ", ".join(result.variables)

        # print cluster
        solutions = "solution"
        if len(result.solutions) != 1:
            solutions += "s"
        return f"{spaces}cluster({variables})[{result.flag}, {len(result.solutions)} {solutions}]\n{s}"
