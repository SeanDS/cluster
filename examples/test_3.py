import logging
import numpy as np

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(name)-25s - %(levelname)-8s - %(message)s"))
logger = logging.getLogger("cluster")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

from cluster import (Problem, GeometricSolver, DistanceConstraint, AngleConstraint, FixConstraint)
from cluster.geometry import Vector
from cluster.view import solution_viewer

problem = Problem()

# Fig 1
problem.add_point('a')
problem.add_point('b')
problem.add_point('c')
problem.add_point('d')
problem.add_point('e')
problem.add_point('f')

problem.add_constraint(AngleConstraint('a', 'b', 'c', np.radians(70)))
problem.add_constraint(AngleConstraint('a', 'd', 'c', np.radians(-70)))
problem.add_constraint(AngleConstraint('a', 'd', 'e', np.radians(70)))
problem.add_constraint(AngleConstraint('a', 'f', 'e', np.radians(-70)))

problem.add_constraint(AngleConstraint('b', 'a', 'c', np.radians(-30)))
problem.add_constraint(AngleConstraint('c', 'a', 'd', np.radians(-30)))
problem.add_constraint(AngleConstraint('d', 'a', 'e', np.radians(-30)))
problem.add_constraint(AngleConstraint('e', 'a', 'f', np.radians(-30)))

problem.add_constraint(DistanceConstraint('b', 'f', 300))

print("problem:")
print(problem)
solver = GeometricSolver(problem)
# print(diagnostic messages for drplan
print("drplan:")
# at this point, the solver has already solved it, if a solution exists
print(solver.solver)
print("number of top-level rigids:",len(list(solver.solver.top_level())))
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

print("Root = " + str(list(solver.solver._graph.successors('_root'))))

solution_viewer(problem, result.solutions[0])

#import networkx as nx
#nx.drawing.nx_pydot.write_dot(solver.solver._mg, "test_3.dot")
#nx.drawing.nx_pydot.write_dot(solver.constraint_graph, "test_3.dot")
