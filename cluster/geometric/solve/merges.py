import abc
import logging
import copy
import numpy as np
import numpy.linalg as la

from ...configuration import Configuration
from ...geometry import Vector, cc_int, cr_int, rr_int, tol_zero
from ...method import Method
from ..constraints import (NotClockwiseConstraint, NotCounterClockwiseConstraint,
                           NotAcuteConstraint, NotObtuseConstraint)

LOGGER = logging.getLogger(__name__)

class Merge(Method, metaclass=abc.ABCMeta):
    """A merge is a method such that a single output cluster satisfies
    all constraints in several input clusters. The output cluster
    replaces the input clusters in the constriant problem"""

    NAME = "Merge"

    def __init__(self, consistent, overconstrained, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.consistent = consistent
        self.overconstrained = overconstrained

    def prototype_constraints(self):
        """Default prototype constraints"""

        # empty list of constraints
        return []

    def __str__(self):
        # get parent string
        string = super().__str__()

        # add status and return
        return f"{string}[{self.status_str()}]"

    def status_str(self):
        if self.consistent:
            consistent_status = "consistent"
        else:
            consistent_status = "inconsistent"

        if self.overconstrained:
            constrained_status = "overconstrained"
        else:
            constrained_status = "well constrained"

        return "{0}, {1}".format(consistent_status, constrained_status)


class MergePR(Merge):
    """Represents a merging of one point with a rigid

    The first cluster determines the orientation of the resulting cluster.
    """

    NAME = "MergePR"

    def __init__(self, in1, in2, out):
        super().__init__(inputs=[in1, in2], outputs=[out], overconstrained=False, consistent=True)

    def multi_execute(self, inmap):
        LOGGER.debug("MergePR.multi_execute called")

        c1 = self.inputs[0]
        c2 = self.inputs[1]

        conf1 = inmap[c1]
        conf2 = inmap[c2]

        if len(c1.variables) == 1:
            return [copy.copy(conf2)]
        else:
            return [copy.copy(conf1)]


class MergeRR(Merge):
    """Represents a merging of two rigids (overconstrained)

    The first rigid determines the orientation of the resulting cluster"""

    NAME = "MergeRR"

    def __init__(self, in1, in2, out):
        super().__init__(inputs=[in1, in2], outputs=[out], overconstrained=True, consistent=True)

    def multi_execute(self, inmap):
        LOGGER.debug("MergeRR.multi_execute called")

        c1 = self.inputs[0]
        c2 = self.inputs[1]

        conf1 = inmap[c1]
        conf2 = inmap[c2]

        return [conf1.merge(conf2)[0]]


class MergeRH(Merge):
    """Represents a merging of a rigid and a hog (where the hog is absorbed by the rigid)

    Overconstrained.
    """

    NAME = "MergeRH"

    def __init__(self, rigid, hog, out):
        super().__init__(inputs=[rigid, hog], outputs=[out], overconstrained=True, consistent=True)

        self.rigid = rigid
        self.hog = hog
        self.output = out

    def multi_execute(self, inmap):
        LOGGER.debug("MergeRH.multi_execute called")

        conf1 = inmap[self.rigid]

        return [copy.copy(conf1)]


class MergeBH(Merge):
    """Represents a merging of a balloon and a hog (where the hog is absorbed by the balloon)

    Overconstrained.
    """

    NAME = "MergeBH"

    def __init__(self, balloon, hog, out):
        super().__init__(inputs=[balloon, hog], outputs=[out], overconstrained=True,
                         consistent=True)

        self.balloon = balloon

    def multi_execute(self, inmap):
        LOGGER.debug("MergeBH.multi_execute called")

        conf1 = inmap[self.balloon]

        return [copy.copy(conf1)]


class MergeRRR(Merge):
    """Represents a merging of three rigids

    The first rigid determines the orientation of the resulting cluster.
    """

    NAME = "MergeRRR"

    def __init__(self, r1, r2, r3, out):
        super().__init__(inputs=[r1, r2, r3], outputs=[out], overconstrained=False, consistent=True)

        # check coincidence
        shared12 = set(r1.variables).intersection(r2.variables)
        shared13 = set(r1.variables).intersection(r3.variables)
        shared23 = set(r2.variables).intersection(r3.variables)
        shared1 = shared12.union(shared13)
        shared2 = shared12.union(shared23)
        shared3 = shared13.union(shared23)

        if len(shared12) < 1:
            raise Exception("underconstrained r1 and r2")
        elif len(shared12) > 1:
            LOGGER.debug("Overconstrained RRR: r1 and r2")

            self.overconstrained = True
        if len(shared13) < 1:
            raise Exception("underconstrained r1 and r3")
        elif len(shared13) > 1:
            LOGGER.debug("Overconstrained RRR: r1 and r3")

            self.overconstrained = True
        if len(shared23) < 1:
            raise Exception("underconstrained r2 and r3")
        elif len(shared23) > 1:
            LOGGER.debug("Overconstrained RRR: r2 and r3")

            self.overconstrained = True
        if len(shared1) < 2:
            raise Exception("underconstrained r1")
        elif len(shared1) > 2:
            LOGGER.debug("Overconstrained RRR: r1")

            self.overconstrained = True
        if len(shared2) < 2:
            raise Exception("underconstrained r2")
        elif len(shared2) > 2:
            LOGGER.debug("Overconstrained RRR: r2")

            self.overconstrained = True
        if len(shared3) < 2:
            raise Exception("underconstrained r3")
        elif len(shared3) > 2:
            LOGGER.debug("Overconstrained RRR: r3")

            self.overconstrained = True

    def multi_execute(self, inmap):
        LOGGER.debug("MergeRRR.multi_execute called")

        r1 = inmap[self.inputs[0]]
        r2 = inmap[self.inputs[1]]
        r3 = inmap[self.inputs[2]]

        shared12 = set(r1.variables).intersection(r2.variables).difference(r3.variables)
        shared13 = set(r1.variables).intersection(r3.variables).difference(r2.variables)
        shared23 = set(r2.variables).intersection(r3.variables).difference(r1.variables)

        v1 = list(shared12)[0]
        v2 = list(shared13)[0]
        v3 = list(shared23)[0]

        assert v1 != v2
        assert v1 != v3
        assert v2 != v3

        p11 = r1.position(v1)
        p21 = r1.position(v2)
        d12 = la.norm(p11 - p21)
        p23 = r3.position(v2)
        p33 = r3.position(v3)
        d23 = la.norm(p23 - p33)
        p32 = r2.position(v3)
        p12 = r2.position(v1)
        d31 = la.norm(p32 - p12)

        ddds = MergeRRR.solve_ddd(v1, v2, v3, d12, d23, d31)

        solutions = []

        for s in ddds:
            solution = r1.merge(s)[0].merge(r2)[0].merge(r3)[0]

            solutions.append(solution)

        return solutions

    def prototype_constraints(self):
        r1 = self.inputs[0]
        r2 = self.inputs[1]
        r3 = self.inputs[2]

        shared12 = set(r1.variables).intersection(r2.variables).difference(r3.variables)
        shared13 = set(r1.variables).intersection(r3.variables).difference(r2.variables)
        shared23 = set(r2.variables).intersection(r3.variables).difference(r1.variables)

        v1 = list(shared12)[0]
        v2 = list(shared13)[0]
        v3 = list(shared23)[0]

        assert v1 != v2
        assert v1 != v3
        assert v2 != v3

        constraints = []

        constraints.append(NotCounterClockwiseConstraint(v1, v2, v3))
        constraints.append(NotClockwiseConstraint(v1, v2, v3))

        return constraints

    @staticmethod
    def solve_ddd(v1, v2, v3, d12, d23, d31):
        LOGGER.debug("Solving ddd: %s %s %s %f %f %f", v1, v2, v3, d12, d23, d31)

        p1 = Vector.origin()
        p2 = Vector([d12, 0.0])
        p3s = cc_int(p1, d31, p2, d23)

        solutions = []

        for p3 in p3s:
            solution = Configuration({v1:p1, v2:p2, v3:p3})

            solutions.append(solution)

        return solutions


class MergeRHR(Merge):
    """Represents a merging of two rigids and a hedgehog

    The first rigid determines the orientation of the resulting cluster
    """

    NAME = "MergeRHR"

    def __init__(self, c1, hog, c2, out):
        super().__init__(inputs=[c1, hog, c2], outputs=[out], overconstrained=False,
                         consistent=True)

        self.c1 = c1
        self.hog = hog
        self.c2 = c2
        self.output = out

        # check coincidence
        if not (hog.cvar in c1.variables and hog.cvar in c2.variables):
            raise Exception("hog.cvar not in c1.variables and c2.variables")

        shared12 = set(c1.variables).intersection(c2.variables)
        shared1h = set(c1.variables).intersection(hog.xvars)
        shared2h = set(c2.variables).intersection(hog.xvars)

        shared1 = shared12.union(shared1h)
        shared2 = shared12.union(shared2h)
        sharedh = shared1h.union(shared2h)

        if len(shared12) < 1:
            raise Exception("underconstrained c1 and c2")
        elif len(shared12) > 1:
            LOGGER.debug("Overconstrained CHC: c1 and c2")

            self.overconstrained = True
        if len(shared1h) < 1:
            raise Exception("underconstrained c1 and hog")
        elif len(shared1h) > 1:
            LOGGER.debug("Overconstrained CHC: c1 and hog")

            self.overconstrained = True
        if len(shared2h) < 1:
            raise Exception("underconstrained c2 and hog")
        elif len(shared2h) > 1:
            LOGGER.debug("Overconstrained CHC: c2 and hog")

            self.overconstrained = True
        if len(shared1) < 2:
            raise Exception("underconstrained c1")
        elif len(shared1) > 2:
            LOGGER.debug("Overconstrained CHC: c1")

            self.overconstrained = True
        if len(shared2) < 2:
            raise Exception("underconstrained c2")
        elif len(shared2) > 2:
            LOGGER.debug("Overconstrained CHC: c2")

            self.overconstrained = True
        if len(sharedh) < 2:
            raise Exception("underconstrained hog")
        elif len(shared1) > 2:
            LOGGER.debug("Overconstrained CHC: hog")

            self.overconstrained = True

    def multi_execute(self, inmap):
        LOGGER.debug("MergeRHR.multi_execute called")

        # determine vars
        shared1 = set(self.hog.xvars).intersection(self.c1.variables)
        shared2 = set(self.hog.xvars).intersection(self.c2.variables)

        v1 = list(shared1)[0]
        v2 = self.hog.cvar
        v3 = list(shared2)[0]

        # get configs
        conf1 = inmap[self.c1]
        confh = inmap[self.hog]
        conf2 = inmap[self.c2]

        # determine angle
        p1h = confh.position(v1)
        p2h = confh.position(v2)
        p3h = confh.position(v3)
        a123 = p2h.angle_between(p1h, p3h)

        # d1c
        p11 = conf1.position(v1)
        p21 = conf1.position(v2)
        d12 = p11.distance_to(p21)

        # d2c
        p32 = conf2.position(v3)
        p22 = conf2.position(v2)
        d23 = p32.distance_to(p22)

        # solve
        dads = MergeRHR.solve_dad(v1, v2, v3, d12, a123, d23)
        solutions = []

        for s in dads:
            solution = conf1.merge(s)[0].merge(conf2)[0]
            solutions.append(solution)

        return solutions

    @staticmethod
    def solve_dad(v1, v2, v3, d12, a123, d23):
        LOGGER.debug("Solving dad: %s %s %s %f %f %f", v1, v2, v3, d12, a123, d23)

        p2 = Vector.origin()
        p1 = Vector([d12, 0.0])
        p3s = [Vector([d23 * np.cos(a123), d23 * np.sin(a123)])]

        solutions = []

        for p3 in p3s:
            solution = Configuration({v1: p1, v2: p2, v3: p3})
            solutions.append(solution)

        return solutions


class MergeRRH(Merge):
    """Represents a merging of two rigids and a hedgehog

    The first rigid determines the orientation of the resulting cluster.
    """

    NAME = "MergeRRH"

    def __init__(self, c1, c2, hog, out):
        super().__init__(inputs=[c1, c2, hog], outputs=[out], overconstrained=False,
                         consistent=True)

        self.c1 = c1
        self.c2 = c2
        self.hog = hog
        self.output = out

        # check coincidence
        if hog.cvar not in c1.variables:
            raise Exception("hog.cvar not in c1.variables")
        if hog.cvar in c2.variables:
            raise Exception("hog.cvar in c2.variables")

        shared12 = set(c1.variables).intersection(c2.variables)
        shared1h = set(c1.variables).intersection(hog.xvars)
        shared2h = set(c2.variables).intersection(hog.xvars)

        shared1 = shared12.union(shared1h)
        shared2 = shared12.union(shared2h)
        sharedh = shared1h.union(shared2h)

        if len(shared12) < 1:
            raise Exception("underconstrained c1 and c2")
        elif len(shared12) > 1:
            LOGGER.debug("Overconstrained CCH: c1 and c2")

            self.overconstrained = True
        if len(shared1h) < 1:
            raise Exception("underconstrained c1 and hog")
        elif len(shared1h) > 1:
            LOGGER.debug("Overconstrained CCH: c1 and hog")

            self.overconstrained = True
        if len(shared2h) < 1:
            raise Exception("underconstrained c2 and hog")
        elif len(shared2h) > 2:
            LOGGER.debug("Overconstrained CCH: c2 and hog")

            self.overconstrained = True
        if len(shared1) < 1:
            raise Exception("underconstrained c1")
        elif len(shared1) > 1:
            LOGGER.debug("Overconstrained CCH: c1")

            self.overconstrained = True
        if len(shared2) < 2:
            raise Exception("underconstrained c2")
        elif len(shared2) > 2:
            LOGGER.debug("Overconstrained CCH: c2")

            self.overconstrained = True
        if len(sharedh) < 2:
            raise Exception("underconstrained hog")
        elif len(sharedh) > 2:
            LOGGER.debug("Overconstrained CCH: hedgehog")

            self.overconstrained = True

    def multi_execute(self, inmap):
        LOGGER.debug("MergeRRH.multi_execute called")

        # assert hog.cvar in c1
        if self.hog.cvar in self.c1.variables:
            c1 = self.c1
            c2 = self.c2
        else:
            c1 = self.c2
            c2 = self.c1

        # get v1
        v1 = self.hog.cvar

        # get v2
        candidates2 = set(self.hog.xvars).intersection(c1.variables).intersection(c2.variables)

        assert len(candidates2) >= 1

        v2 = list(candidates2)[0]

        # get v3
        candidates3 = set(self.hog.xvars).intersection(c2.variables).difference([v1, v2])

        assert len(candidates3) >= 1

        v3 = list(candidates3)[0]

        # check
        assert v1 != v2
        assert v1 != v3
        assert v2 != v3

        # get configs
        confh = inmap[self.hog]
        conf1 = inmap[c1]
        conf2 = inmap[c2]

        # get angle
        p1h = confh.position(v1)
        p2h = confh.position(v2)
        p3h = confh.position(v3)
        a312 = p1h.angle_between(p3h, p2h)

        # get distance d12
        p11 = conf1.position(v1)
        p21 = conf1.position(v2)
        d12 = p11.distance_to(p21)

        # get distance d23
        p22 = conf2.position(v2)
        p32 = conf2.position(v3)
        d23 = p22.distance_to(p32)
        adds = MergeRRH.solve_add(v1, v2, v3, a312, d12, d23)

        solutions = []

        # do merge (note, order c1 c2 restored)
        conf1 = inmap[self.c1]
        conf2 = inmap[self.c2]

        for s in adds:
            solution = conf1.merge(s)[0].merge(conf2)[0]
            solutions.append(solution)

        return solutions

    @staticmethod
    def solve_add(a,b,c, a_cab, d_ab, d_bc):
        LOGGER.debug("Solving add: %s %s %s %f %f %f", a, b, c, a_cab, d_ab, d_bc)

        p_a = Vector.origin()
        p_b = Vector([d_ab, 0.0])

        direction = Vector([np.cos(-a_cab), np.sin(-a_cab)])

        solutions = cr_int(p_b, d_bc, p_a, direction)

        rval = []

        for s in solutions:
            p_c = s

            rval.append(Configuration({a: p_a, b: p_b, c: p_c}))

        return rval

    def prototype_constraints(self):
        # assert hog.cvar in c1
        if self.hog.cvar in self.c1.variables:
            c1 = self.c1
            c2 = self.c2
        else:
            c1 = self.c2
            c2 = self.c1

        shared1h = set(self.hog.xvars).intersection(c1.variables).difference([self.hog.cvar])
        shared2h = set(self.hog.xvars).intersection(c2.variables).difference(shared1h)

        # get vars
        v1 = self.hog.cvar
        v2 = list(shared1h)[0]
        v3 = list(shared2h)[0]

        assert v1 != v2
        assert v1 != v3
        assert v2 != v3

        constraints = []

        constraints.append(NotAcuteConstraint(v2, v3, v1))
        constraints.append(NotObtuseConstraint(v2, v3, v1))

        return constraints


class BalloonFromHogs(Merge):
    """Represent a balloon merged from two hogs"""

    NAME = "BalloonFromHedgehogs"

    def __init__(self, hog1, hog2, balloon):
        """Create a new balloon from two angles

           keyword args:
            hog1 - a Hedghog
            hog2 - a Hedehog
            balloon - a Balloon instance
        """

        super().__init__(inputs=[hog1, hog2], outputs=[balloon], overconstrained=False,
                         consistent=True)

        self.hog1 = hog1
        self.hog2 = hog2
        self.balloon = balloon

        # check coincidence
        if hog1.cvar == hog2.cvar:
            raise Exception("hog1.cvar is hog2.cvar")

        shared12 = set(hog1.xvars).intersection(hog2.xvars)

        if len(shared12) < 1:
            raise Exception("underconstrained")

    def multi_execute(self, inmap):
        LOGGER.debug("BalloonFromHogs.multi_execute called")

        v1 = self.hog1.cvar
        v2 = self.hog2.cvar

        shared = set(self.hog1.xvars).intersection(self.hog2.xvars).difference([v1,v2])

        v3 = list(shared)[0]

        assert v1 != v2
        assert v1 != v3
        assert v2 != v3

        # determine angle312
        conf1 = inmap[self.hog1]

        p31 = conf1.position(v3)
        p11 = conf1.position(v1)
        p21 = conf1.position(v2)
        a312 = p11.angle_between(p31, p21)

        # determine distance d12
        d12 = 1.0

        # determine angle123
        conf2 = inmap[self.hog2]
        p12 = conf2.position(v1)
        p22 = conf2.position(v2)
        p32 = conf2.position(v3)
        a123 = p22.angle_between(p12, p32)

        # solve
        return BalloonFromHogs.solve_ada(v1, v2, v3, a312, d12, a123)

    @staticmethod
    def solve_ada(a, b, c, a_cab, d_ab, a_abc):
        LOGGER.debug("Solve ada: %s %s %s %f %f %f", a, b, c, a_cab, d_ab, a_abc)

        p_a = Vector.origin()
        p_b = Vector([d_ab, 0.0])

        dir_ac = Vector([np.cos(-a_cab), np.sin(-a_cab)])
        dir_bc = Vector([-np.cos(-a_abc), np.sin(-a_abc)])

        if tol_zero(np.sin(a_cab)) and tol_zero(np.sin(a_abc)):
            m = d_ab / 2 + np.cos(-a_cab) * d_ab - np.cos(-a_abc) * d_ab

            p_c = Vector([m, 0.0])

            mapping = {a: p_a, b: p_b, c: p_c}

            cluster = Configuration(mapping)
            cluster.underconstrained = True

            rval = [cluster]
        else:
            solutions = rr_int(p_a, dir_ac, p_b, dir_bc)

            rval = []

            for s in solutions:
                p_c = s
                mapping = {a: p_a, b: p_b, c: p_c}

                rval.append(Configuration(mapping))

        return rval


class BalloonMerge(Merge):
    """Represents a merging of two balloons"""

    NAME = "BalloonMerge"

    def __init__(self, in1, in2, out):
        super().__init__(inputs=[in1, in2], outputs=[out], overconstrained=False, consistent=True)

        self.input1 = in1
        self.input2 = in2
        self.output = out
        self.shared = list(set(self.input1.variables).intersection(self.input2.variables))

        shared = set(in1.variables).intersection(in2.variables)

        if len(shared) < 2:
            raise Exception("underconstrained")
        elif len(shared) > 2:
            LOGGER.debug("Overconstrained balloon merge")

            self.overconstrained = True

    def multi_execute(self, inmap):
        LOGGER.debug("BalloonMerge.multi_execute called")

        c1 = self.inputs[0]
        c2 = self.inputs[1]

        conf1 = inmap[c1]
        conf2 = inmap[c2]

        return [conf1.merge_scale(conf2)[0]]


class BalloonRigidMerge(Merge):
    """Represents a merging of a balloon and a rigid"""

    NAME = "BalloonRigidMerge"

    def __init__(self, balloon, cluster, output):
        super().__init__(inputs=[balloon, cluster], outputs=[output], overconstrained=False,
                         consistent=True)

        self.balloon = balloon
        self.cluster= cluster

        # FIXME: is this used?
        self.shared = list(set(self.balloon.variables).intersection(self.cluster.variables))

        # check coincidence
        shared = set(balloon.variables).intersection(cluster.variables)

        if len(shared) < 2:
            raise Exception("underconstrained balloon-cluster merge")
        elif len(shared) > 2:
            LOGGER.debug("Overconstrained merge of %s and %s", balloon, cluster)

            self.overconstrained = True

    def multi_execute(self, inmap):
        LOGGER.debug("BalloonRigidMerge.multi_execute called")

        rigid = inmap[self.cluster]
        balloon = inmap[self.balloon]

        return [rigid.merge_scale(balloon)[0]]


class MergeHogs(Merge):
    """Represents a merging of two hogs to form a new hog"""

    NAME = "MergeHogs"

    def __init__(self, hog1, hog2, output):
        super().__init__(inputs=[hog1, hog2], outputs=[output], overconstrained=False,
                         consistent=True)

        self.hog1 = hog1
        self.hog2 = hog2
        self.output = output

        if hog1.cvar != hog2.cvar:
            raise Exception("hog1.cvar != hog2.cvar")

        shared = set(hog1.xvars).intersection(hog2.xvars)

        if len(shared) < 1:
            raise Exception("underconstrained balloon-cluster merge")
        elif len(shared) > 1:
            LOGGER.debug("Overconstrained merge of %s and %s", hog1, hog2)

            self.overconstrained = True

    def multi_execute(self, inmap):
        LOGGER.debug("MergeHogs.multi_execute called")

        conf1 = inmap[self.inputs[0]]
        conf2 = inmap[self.inputs[1]]

        shared = set(self.hog1.xvars).intersection(self.hog2.xvars)

        conf12 = conf1.merge_scale(conf2, [self.hog1.cvar, list(shared)[0]])[0]

        return [conf12]
