"""Base classes for multi-valued assignments in methodgraphs"""

import abc
import logging

from .graph import MethodGraph

LOGGER = logging.getLogger(__name__)

class Variable(object):
    """Represents multi-valued variables"""

    def __init__(self, name=None):
        self.name = name

    def __str__(self):
        if self.name is None:
            return "Variable#{0}".format(id(self))

        return "Variable({0})".format(self.name)

    def __repr__(self):
        return str(self)


class Method(metaclass=abc.ABCMeta):
    """Method that is executed for multiple alternative inputs, resulting in multiple output values.

    Input may optionally contain Variable instances.
    There must be a single Variable output variable.

    Subclasses should implement the 'multi_execute' method, not overide the \
    'execute' method. This method is called for every permutation of values of \
    multi-valued input variables.

    Any input variables that are instances of Variable will be \
    replaced by their shadowed counterpart in the input map for multi_execute.

    The 'multi_execute' method must return a list of possible values for the \
    output variable. The output values returned by subsequent calls \
    multi-execute are collected and stored in the output \
    Variable.
    """

    def __init__(self, name, inputs, outputs):
        self.name = str(name)
        self.inputs = list(inputs)
        self.outputs = list(outputs)

        # empty list of multi inputs
        self.multi_inputs = []

        for variable in self.inputs:
            if isinstance(variable, Variable):
                self.multi_inputs.append(variable)

        if len(self.outputs) != 1:
            raise Exception("Method requires exactly one output")

        if not isinstance(self.outputs[0], Variable):
            raise Exception("Method requires a Variable output")

    def execute(self, inmap):
        """Calls multi_execute for each permutation of multi-valued input \
        variables and collects result in multi-valued ouput variables

        Subclasses should implement multi_execute.
        """

        base_inmap = {}

        for variable in self.inputs:
            if variable not in self.multi_inputs:
                value = inmap[variable]
                base_inmap[variable] = value

        outvar = self.outputs[0]
        values = self._recurse_execute(inmap, base_inmap, self.multi_inputs)

        return {outvar: values}

    @abc.abstractmethod
    def multi_execute(self, inmap):
        raise NotImplementedError

    def _recurse_execute(self, inmap, base_inmap, multi_inputs):
        if len(multi_inputs) > 0:
            mvar = multi_inputs[0]
            values = inmap[mvar]
            output = set()

            for value in values:
                base_inmap[mvar] = value
                output.update(self._recurse_execute(inmap, base_inmap, \
                multi_inputs[1:]))

            return output
        else:
            return self.multi_execute(base_inmap)

    def __str__(self):
        # comma separated list of inputs
        input_str = " + ".join([str(_input) for _input in self.inputs])

        # comma separated list of outputs
        output_str = " + ".join([str(output) for output in self.outputs])

        # combined string
        return "{0}({1} -> {2})".format(self.name, input_str, \
        output_str)


class SumProdMethod(Method):
    """A Method that assigns the sum and product of its input to its output
    Variable"""

    def __init__(self, a, b, c):
        super(SumProdMethod, self).__init__("SumProdMethod", [a, b], [c])

    def multi_execute(self, inmap):
        a = inmap[self.inputs[0]]
        b = inmap[self.inputs[1]]

        return [a + b, a * b]
