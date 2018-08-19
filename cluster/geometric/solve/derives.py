import abc
import logging

from ...method import Method

LOGGER = logging.getLogger(__name__)

class Derive(Method, metaclass=abc.ABCMeta):
    """A derive is a method such that a single output cluster is a
    subconstraint of a single input cluster."""
    NAME = "Derive"

class RigidToHog(Derive):
    """Represents a derivation of a hog from a cluster"""

    NAME = "RigidToHedgehog"

    def __init__(self, cluster, hog):
        super().__init__(inputs=[cluster], outputs=[hog])

        self.cluster = cluster
        self.hog = hog

    def multi_execute(self, inmap):
        LOGGER.debug("RigidToHog.multi_execute called")

        conf1 = inmap[self.inputs[0]]
        variables = list(self.outputs[0].xvars) + [self.outputs[0].cvar]
        conf = conf1.select(variables)

        return [conf]

class BalloonToHog(Derive):
    """Represents a derivation of a hog from a balloon"""

    NAME = "BalloonToHedgehog"

    def __init__(self, balloon, hog):
        super().__init__(inputs=[balloon], outputs=[hog])
        self.balloon = balloon
        self.hog = hog

    def multi_execute(self, inmap):
        LOGGER.debug("BalloonToHog.multi_execute called")

        conf1 = inmap[self.inputs[0]]
        variables = list(self.outputs[0].xvars) + [self.outputs[0].cvar]
        conf = conf1.select(variables)

        return [conf]

class SubHog(Derive):
    NAME = "HedgehogToHedgehog"

    def __init__(self, hog, sub):
        super(SubHog, self).__init__(inputs=[hog], outputs=[sub])

        self.hog = hog
        self.sub = sub

    def multi_execute(self, inmap):
        LOGGER.debug("SubHog.multi_execute called")

        conf1 = inmap[self.inputs[0]]
        variables = list(self.outputs[0].xvars) + [self.outputs[0].cvar]
        conf = conf1.select(variables)

        return [conf]
