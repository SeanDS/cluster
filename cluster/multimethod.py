"""Base classes for multi-valued assignments in methodgraphs"""

import abc
from .method import Method

class MultiVariable(metaclass=abc.ABCMeta):
    """For representing multi-valued variables"""

    NAME = "Variable"

    # last unique variable id
    _last_id = 0

    # unique number
    _number = None

    def __new__(cls, *args, **kwargs):
        # create object
        obj = super().__new__(cls)

        # assign unique variable ID
        obj._number = cls._last_id

        # increment last id
        cls._last_id += 1

        return obj

    def __str__(self):
        return f"{self.NAME}#{self._number}"

    def __repr__(self):
        return str(self)


class MultiMethod(Method):
    """A Method that is executed for multiple alternative inputs, resulting
       in multiple output values.

       Input may optionally contain MultiVariable instances.
       There must be a single MultiVariable output variable

       Subclasses should implement the 'multi_execute' method, not overide the 'execute' method.
       This method is called for every permutation of values of multi-valued input variables.

       Any input variables that are instances of MultiVariable will be replaced by their
       shadowed counterpart in the input map for multi_execute.

       The 'multi_execute' method must return a list of possible values for the output variable.
       The output values returned by subsequent calls multi-execute are collected and stored in the
       output MultiVariable.

       Note that a set of values for the outputvariable is stored, so that equivalent values are only stored once.
    """

    def __init__(self):
        """Call this initialize after _inputs and _outputs has been set"""
        self._multi_inputs = []
        for variable in self._inputs:
            if isinstance(variable, MultiVariable):
                self._multi_inputs.append(variable)
        if len(self._outputs) != 1:
            raise Exception("requires exactly one output")
        if not isinstance(self._outputs[0], MultiVariable):
            raise Exception("requires a MultiVariable output")


    def execute(self, inmap):
        """calls multi_execute for each permutation of multi-valued input variables and collects
           result in multi-valued ouput variables. Subclasses should implement multi_execute."""
        base_inmap = {}
        for variable in self._inputs:
            if variable not in self._multi_inputs:
                value = inmap[variable]
                base_inmap[variable] = value

        outvar = self._outputs[0]
        values = self._recurse_execute(inmap, base_inmap, self._multi_inputs)
        return {outvar:values}

    def _recurse_execute(self, inmap, base_inmap, multi_inputs):
        if len(multi_inputs) > 0:
            mvar = multi_inputs[0]
            values = inmap[mvar]
            output = set()
            for value in values:
                base_inmap[mvar] = value
                output.update(self._recurse_execute(inmap, base_inmap, multi_inputs[1:]))
            return output
        else:
            return self.multi_execute(base_inmap)


#####

class SumProdMethod(MultiMethod):
    """A MultiMethod that assigns the sum and product of its input to it's ouput MultiVariable"""

    NAME = "SumProdMethod"

    def __init__(self, a,b,c):
        self._inputs = [a,b]
        self._outputs = [c]
        MultiMethod.__init__(self)

    def multi_execute(self, inmap):
        #print str(inmap)
        a = inmap[self._inputs[0]]
        b = inmap[self._inputs[1]]
        result = [a+b, a*b]
        #print result
        return result

def test():
    from .methodgraph import MethodGraph
    graph = MethodGraph()
    graph.add_variable('a', 1)
    graph.add_variable('b', 2)
    mv_x = MultiVariable('x')
    graph.add_variable(mv_x)
    graph.add_method(SumProdMethod('a','b', mv_x))

    graph.add_variable('p', 3)
    graph.add_variable('q', 4)
    mv_y = MultiVariable('y')
    graph.add_variable(mv_y)
    graph.add_method(SumProdMethod('p','q', mv_y))

    mv_z = MultiVariable('z')
    graph.add_variable(mv_z)
    graph.add_method(SumProdMethod(mv_x,mv_y,mv_z))

    print(graph.get(mv_z))

    graph.set('a', 100)
    print(graph.get(mv_z))


if __name__== '__main__': test()


