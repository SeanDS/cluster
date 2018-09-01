import logging

from ..configuration import Configuration
from .base import Method

LOGGER = logging.getLogger(__name__)

class PrototypeMethod(Method):
    """Selects solutions of a cluster for which the prototype and the solution satisfy the same constraints."""

    NAME = "PrototypeMethod"

    def __init__(self, incluster, selclusters, outcluster, constraints):
        # call parent constructor
        super().__init__(inputs=[incluster]+selclusters, outputs=[outcluster])

        # set constraints
        self.constraints = list(constraints)

    def multi_execute(self, inmap):
        LOGGER.debug("PrototypeMethod.multi_execute called")

        incluster = self.inputs[0]
        selclusters = []

        for i in range(1, len(self.inputs)):
            selclusters.append(self.inputs[i])

        LOGGER.debug("Input clusters: %s", incluster)
        LOGGER.debug("Selection clusters: %s", selclusters)

        # get confs
        inconf = inmap[incluster]
        selmap = {}

        for cluster in selclusters:
            conf = inmap[cluster]

            assert len(conf.variables) == 1

            var = list(conf.variables)[0]
            selmap[var] = conf.mapping[var]

        selconf = Configuration(selmap)
        sat = True

        LOGGER.debug("Input configuration: %s", inconf)
        LOGGER.debug("Selection configuration: %s", selconf)

        for con in self.constraints:
            satcon = con.satisfied(inconf.mapping) != con.satisfied(selconf.mapping)

            LOGGER.debug("Constraint: %s", con)
            LOGGER.debug("Constraint satisfied? %s", satcon)
            sat = sat and satcon

        LOGGER.debug("Prototype satisfied? %s", sat)

        if sat:
            return [inconf]

        return []
