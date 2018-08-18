"""Module for method graphs.

A method graph contains variables and methods. Methods are objects that specify
input and output variables and an 'execute' method. Whenever the value of a
variable is changed, one or more methods are executed to update the value of
'upstream' variables."""

import abc
import logging

LOGGER = logging.getLogger(__name__)

class Method(object, metaclass=abc.ABCMeta):
    """Defines input variables, output variables and an execute method

    Instances must be immutable, hashable objects.
    """

    def __init__(self, name, inputs, outputs):
        """Instantiates a new Method"""

        self.name = str(name)
        self.inputs = list(inputs)
        self.outputs = list(outputs)

    @abc.abstractmethod
    def execute(self, inmap):
        """Execute method

        Returns a mapping (dict) of output variables to values, given an input
        map that maps input variables to values (dict). The previous value of
        the output variable should also be in inmap. If the method cannot be
        executed, it should return an empty map.
        """

        raise NotImplementedError()

    def __str__(self):
        # comma separated list of inputs
        input_str = " + ".join([str(_input) for _input in self.inputs])

        # comma separated list of outputs
        output_str = " + ".join([str(output) for output in self.outputs])

        # combined string
        return "{0}({1} -> {2})".format(self.name, input_str, \
        output_str)

class AddMethod(Method):
    """Method representing addition of two variables"""

    def __init__(self, a, b, c):
        """Instantiates a new AddMethod

        :param a: first input
        :param b: second input
        :param c: output
        """

        # call parent to set appropriate inputs and outputs
        super(AddMethod, self).__init__("AddMethod", inputs=[a, b], outputs=[c])

    def execute(self, in_map):
        """Execute method"""

        # map of outputs to input values
        out_map = {}

        # get variables from lists of inputs and outputs
        a = self.inputs[0]
        b = self.inputs[1]
        c = self.outputs[0]

        # calculate c = a + b if the variables exist in the input map
        if a in in_map and b in in_map \
        and in_map[a] != None and in_map[b] != None:
            # set the value in the output map
            out_map[c] = in_map[a] + in_map[b]

        return out_map

class SubMethod(Method):
    """Method representing subtraction of two variables"""

    def __init__(self, a, b, c):
        """Instantiates a new SubMethod

        :param a: first input
        :param b: second input
        :param c: output
        """

        # call parent to set appropriate inputs and outputs
        super(SubMethod, self).__init__("SubMethod", inputs=[a, b], outputs=[c])

    def execute(self, in_map):
        """Execute method"""

        # map of outputs to input values
        out_map = {}

        # get variables from lists of inputs and outputs
        a = self.inputs[0]
        b = self.inputs[1]
        c = self.outputs[0]

        # calculate c = a - b if the variables exist in the input map
        if a in in_map and b in in_map \
        and in_map[a] != None and in_map[b] != None:
            # set the value in the output map
            out_map[c] = in_map[a] - in_map[b]

        return out_map

class SetMethod(Method):
    """Method representing the setting of a variable's value"""

    def __init__(self, variable, value):
        """Instantiates a new SetMethod

        :param variable: variable name
        :param value: any object to be associated with var
        """

        # call parent to set appropriate inputs and outputs
        super(SetMethod, self).__init__("SetMethod", inputs=[], \
        outputs=[variable])

        # make a record of the value to be set
        self._value = value

    def execute(self, in_map):
        """Execute method"""

        # return a dict with the output set to the value
        return {self._outputs[0]: self._value}

    def __str__(self):
        """Unicode representation of the method

        Overrides :class:`~.Method`
        """

        # show the output's set value in the string representation
        return "{0}({1}={2})".format(self.name, self._outputs[0], self._value)

class AssignMethod(Method):
    """Method representing the assignment of a value to a variable"""

    def __init__(self, a, b):
        """Instantiates a new AssignMethod

        :param a: first input
        :param b: second input
        """

        # call parent to set appropriate inputs and outputs
        super(AssignMethod, self).__init__("AssignMethod", inputs=[b], \
        outputs=[a])

    def execute(self, in_map):
        """Execute method"""

        # return empty dict of the only input is not in the map
        if self._inputs[0] not in in_map:
            return {}

        # set the relevant output to the mapping and return
        return {self._outputs[0]: in_map(self._inputs[0])}

    def __str__(self):
        # show the input's assigned value in the string representation
        return "{0}({1}={2})".format(self.name, self._inputs[0], self._value)
