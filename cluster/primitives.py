class GeometricVariable:
    """Abstract base class for geometric variabes (Point, Line, etc)
        A geometric variable is identified by its name attibute and its type.
        It is hasable so it can be used in sets etc.
    """

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.__class__.__name__+"("+repr(self.name)+")"


class Point(GeometricVariable):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "Point("+str(self.name)+")"

class Line(GeometricVariable):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return "Line("+str(self.name)+")"
