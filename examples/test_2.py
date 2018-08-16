import logging
import numpy as np

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(name)-25s - %(levelname)-8s - %(message)s"))
logger = logging.getLogger("cluster")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

from cluster.geometric import *
from cluster.geometry import Vector
from cluster.view import solution_viewer

problem = GeometricProblem()

problem.add_point('l1', Vector.origin())
problem.add_point('m1', Vector.origin())
problem.add_point('bs1', Vector.origin())
problem.add_point('m2', Vector.origin())
problem.add_point('m3', Vector.origin())
problem.add_point('m4', Vector.origin())
problem.add_point('m5', Vector.origin())

# \/ laser
# \       m3  /--\ m2    / m5
# |                      |
# \------------/---------/ m4
# m1          bs1

problem.add_constraint(DistanceConstraint('l1', 'm1', 50))
problem.add_constraint(DistanceConstraint('m1', 'bs1', 100))
problem.add_constraint(DistanceConstraint('bs1', 'm2', 30))
problem.add_constraint(DistanceConstraint('bs1', 'm4', 100))
problem.add_constraint(DistanceConstraint('m4', 'm5', 50))

problem.add_constraint(AngleConstraint('l1', 'm1', 'bs1', np.radians(-90)))
problem.add_constraint(AngleConstraint('bs1', 'm4', 'm5', np.radians(-90)))

# triangular cavity
problem.add_constraint(AngleConstraint('bs1', 'm2', 'm3', np.radians(60)))
problem.add_constraint(AngleConstraint('m2', 'm3', 'bs1', np.radians(60)))

# set global orientation of triangular cavity
problem.add_constraint(AngleConstraint('m1', 'bs1', 'm2', np.radians(120)))
problem.add_constraint(AngleConstraint('m1', 'bs1', 'm4', np.radians(180)))

# add an overconstraint
problem.add_constraint(AngleConstraint('m4', 'bs1', 'm3', np.radians(-120)))

print("problem:")
print(problem)
solver = GeometricSolver(problem)
# print(diagnostic messages for drplan
print("drplan:")
# at this point, the solver has already solved it, if a solution exists
print(solver.solver)
print("number of top-level rigids:",len(solver.solver.top_level()))
result = solver.decomposition()
print("result:")
print(result)
print("result is",result.flag, "with", len(result.solutions),"solutions")
check = True
if len(result.solutions) == 0:
    check = False
for sol in result.solutions:
    print("solution:",sol)
    check = check and problem.verify(sol)
if check:
    print("all solutions valid")
else:
    print("INVALID")

solution_viewer(problem, result.solutions[0])
