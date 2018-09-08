"""
Module for method graphs
Copyright Rick van der Meiden, 2003, 2004
Created: 1 Nov 2003.

A method graph contains variables and methods. Methods are objects that
specify input and output variables and an 'execute' method. Whenever the
value of a variable is changed, one or more methods are executed to update
the value of 'upstream' variables.

Changes:
23 Nov 2004 - added Error classes, updated naming and doc conventions (PEP 8, 257)
"""

import abc

# ----------- Exceptions -----------

class ValidityError(Exception):
    """Error indicating operation violated MethodGraph validity"""

    def __init__(self, message):
        """Constructor for ValidityError

           arguments:
               message - message to be displayed
        """
        self._message = message

    def __str__(self):
        return "ValidityError: " + self._message

# ----------- class Method -----------

class Method(metaclass=abc.ABCMeta):
    """Abstract method

       A Method is an object that defines input variables, output variables
       and an execute method. This class should be considered abstract.
       Instances (of subclasses of) Method must be non-mutable, hashable objects.
    """

    NAME = "Method"

    def inputs(self):
        """return a list of input variables

           If an attribute '_inputs' has been defined, a new list
           with the contents of that attribute will be returned.
           Subclasses may choose to initialise this variable or to
           override this function.
        """
        if hasattr(self, "_inputs"):
            return list(self._inputs)
        else:
            raise NotImplementedError

    def outputs(self):
        """return a list of output variables

           If an attribute '_outputs' has been defined, a new list
           with the contents of that attribute will be returned.
           Subclasses may choose to initialise this variable or to
           override this function.
        """
        if hasattr(self, "_outputs"):
            return list(self._outputs)
        else:
            raise NotImplementedError

    def execute(self, inmap):
        """Execute method.

        Returns a mapping (dictionary) of output variables to values,
        given an input map, mapping input variables to values (dictionary)
        The previous value of the output variable should also be in inmap.
        If the method cannot be executed, it should return an empty map.
        """
        raise NotImplementedError

    def __str__(self):
        # comma separated list of inputs
        input_str = " + ".join([str(_input) for _input in self.inputs()])

        # comma separated list of outputs
        output_str = " + ".join([str(output) for output in self.outputs()])

        # combined string
        return f"{self.NAME}({input_str} -> {output_str})"

# ----------- various Methods ---------

class OrMethod(Method):
    NAME = "OrMethod"

    def __init__(self, inputs, output):
        """new method output := input[0] | input[1] | ... """
        self._inputs = list(inputs)
        self._outputs = [output]

    def execute(self, inmap):
        result = False
        for input in self._inputs:
            result = result | inmap[input]
        outmap = {}
        outmap[self._outputs[0]] = result
        return outmap

class AddMethod(Method):
    NAME = "AddMethod"

    def __init__(self, a, b, c):
        """new method c := a + b"""
        self._inputs = [a,b]
        self._outputs = [c]

    def execute(self, inmap):
        outmap = {}
        a = self._inputs[0]
        b = self._inputs[1]
        c = self._outputs[0]
        if a in inmap and b in inmap and \
           inmap[a] != None and inmap[b] != None:
            outmap[c] = inmap[a] + inmap[b]
        #fi
        return outmap

class SubMethod(Method):
    NAME = "SubMethod"

    def __init__(self, a, b, c):
        """new method c := a - b"""
        self._inputs = [a,b]
        self._outputs = [c]

    def execute(self, inmap):
        outmap = {}
        a = self._inputs[0]
        b = self._inputs[1]
        c = self._outputs[0]
        if a in inmap and b in inmap and \
           inmap[a] != None and inmap[b] != None:
            outmap[c] = inmap[a] - inmap[b]
        #fi
        return outmap

class SetMethod(Method):
    NAME = "SetMethod"

    def __init__(self, var, value):
        """new method var := value

           keyword arguments:
               var - variable name
               value - any object to be associated with var

        """
        self._inputs = []
        self._outputs = [var]
        self._value = value

    def execute(self, inmap):
        return {self._outputs[0]:self._value}

class AssignMethod(Method):
    NAME = "AssignMethod"

    def __init__(self, a, b):
        """new method a := b

           keyword arguments:
               a - variable name
               b - variable name
        """
        self._inputs = [b]
        self._outputs = [a]

    def execute(self, inmap):
        if self._inputs[0] in inmap:
           return {self._outputs[0]:inmap(self._inputs[0])}
        else:
           return {}


# ---------- test ----------

def test():
    print("-- testing method graph")
    mg = MethodGraph()
    print("set a = 3")
    mg.add_variable('a', 3)
    print("set b = 4")
    mg.add_variable('b', 4)
    print("c := a + b")
    mg.add_method(AddMethod('a','b','c'))
    print("c = "+str(mg.get('c')))
    print("set a = 10")
    mg.set_node_value('a', 10)
    print("c = "+str(mg.get('c')))
    mg.add_method(AddMethod('a','c','d'))
    print("d := a + c")
    mg.add_method(AddMethod('b','d','e'))
    print("e := b + d")
    print("d = "+str(mg.get('d')))
    print("e = "+str(mg.get('e')))
    print("a := d + e")
    import sys, traceback
    try:
        mg.add_method(AddMethod('d','e','a'))
        print("success: should not be possible")
    except Exception as e:
        print(e)
    print("e := a + b")
    try:
        mg.add_method(AddMethod('a','b','e'))
        print("success: should not be possible")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    test()


