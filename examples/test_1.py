import logging
import numpy as np

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(name)-25s - %(levelname)-8s - %(message)s"))
logger = logging.getLogger("cluster")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

from cluster import (GeometricProblem, GeometricSolver, DistanceConstraint, AngleConstraint,
                     FixConstraint)
from cluster.geometry import Vector
from cluster.view import solution_viewer

problem = GeometricProblem()

problem.add_variable('l1')
problem.add_variable('m1')
problem.add_variable('bs1')
problem.add_variable('m2')
problem.add_variable('m3')
problem.add_variable('m4')
problem.add_variable('m5')

problem.add_constraint(DistanceConstraint('l1', 'm1', 50))
problem.add_constraint(DistanceConstraint('m1', 'bs1', 100))
problem.add_constraint(DistanceConstraint('bs1', 'm2', 50))
problem.add_constraint(DistanceConstraint('bs1', 'm3', 150))
problem.add_constraint(DistanceConstraint('bs1', 'm4', 100))
problem.add_constraint(DistanceConstraint('m2', 'm5', 100))
problem.add_constraint(AngleConstraint('l1', 'm1', 'bs1', np.radians(-90)))
problem.add_constraint(AngleConstraint('m1', 'bs1', 'm2', np.radians(-90)))
problem.add_constraint(AngleConstraint('m4', 'bs1', 'm1', np.radians(-90)))
problem.add_constraint(AngleConstraint('bs1', 'm2', 'm5', np.radians(90)))
problem.add_constraint(AngleConstraint('m3', 'bs1', 'm4', np.radians(-90)))

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
#nx.drawing.nx_pydot.write_dot(solver.solver._mg, "test_1.dot")
#nx.drawing.nx_pydot.write_dot(solver.constraint_graph, "test_1.dot")
