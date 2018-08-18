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

# Advanced LIGO optical layout, D0902838-v5

problem.add_point('laser', Vector.origin())
problem.add_point('sm1', Vector.origin())
problem.add_point('sm2', Vector.origin())
problem.add_point('mc1bk', Vector.origin())
problem.add_point('mc1fr', Vector.origin())
problem.add_point('mc2', Vector.origin())
problem.add_point('mc3fr', Vector.origin())
problem.add_point('mc3bk', Vector.origin())
problem.add_point('im1', Vector.origin())
problem.add_point('im2', Vector.origin())
problem.add_point('im3', Vector.origin())
problem.add_point('im4', Vector.origin())
problem.add_point('prmbk', Vector.origin())
problem.add_point('prmfr', Vector.origin())
problem.add_point('pr2', Vector.origin())
problem.add_point('pr3', Vector.origin())
problem.add_point('bsfr', Vector.origin())
problem.add_point('bsbk', Vector.origin())
problem.add_point('itmy', Vector.origin())
problem.add_point('etmy', Vector.origin())
problem.add_point('itmx', Vector.origin())
problem.add_point('etmx', Vector.origin())
problem.add_point('sr3', Vector.origin())
problem.add_point('sr2', Vector.origin())
problem.add_point('srm', Vector.origin())
problem.add_point('om1', Vector.origin())
problem.add_point('om2', Vector.origin())
problem.add_point('om3', Vector.origin())
problem.add_point('omc1', Vector.origin())
problem.add_point('omc2', Vector.origin())
problem.add_point('omc3', Vector.origin())
problem.add_point('omc4', Vector.origin())
problem.add_point('omc5', Vector.origin())

mc1_aoi = 160
mc1_light_travel_path = 15
mc3_light_travel_path = 15
imc_opening_angle = 85
prm_light_travel_path = 10
l_arm = 100
bs_light_travel_path = 10
bs_aoi = -150

problem.add_constraint(DistanceConstraint('laser', 'sm1', 100))
problem.add_constraint(AngleConstraint('laser', 'sm1', 'sm2', np.radians(45)))
problem.add_constraint(DistanceConstraint('sm1', 'sm2', 25))
problem.add_constraint(DistanceConstraint('sm2', 'mc1bk', 25))
problem.add_constraint(AngleConstraint('sm1', 'sm2', 'mc1bk', np.radians(-50)))
problem.add_constraint(DistanceConstraint('mc1bk', 'mc1fr', mc1_light_travel_path))
problem.add_constraint(AngleConstraint('sm2', 'mc1bk', 'mc1fr', np.radians(mc1_aoi)))
problem.add_constraint(AngleConstraint('mc1bk', 'mc1fr', 'mc2', np.radians(-mc1_aoi)))

# IMC angles together with mc1->mc3 length set the length length
problem.add_constraint(DistanceConstraint('mc1fr', 'mc3fr', 25))
problem.add_constraint(AngleConstraint('mc1fr', 'mc3fr', 'mc2', np.radians(-imc_opening_angle)))
problem.add_constraint(AngleConstraint('mc3fr', 'mc1fr', 'mc2', np.radians(imc_opening_angle)))

problem.add_constraint(AngleConstraint('mc2', 'mc3fr', 'mc3bk', np.radians(-mc1_aoi)))
problem.add_constraint(DistanceConstraint('mc3fr', 'mc3bk', mc3_light_travel_path))
problem.add_constraint(AngleConstraint('mc3fr', 'mc3bk', 'im1', np.radians(mc1_aoi)))
problem.add_constraint(DistanceConstraint('mc3bk', 'im1', 50))
problem.add_constraint(AngleConstraint('mc3bk', 'im1', 'im2', np.radians(100)))
problem.add_constraint(DistanceConstraint('im1', 'im2', 100))
problem.add_constraint(DistanceConstraint('im2', 'im3', 90))
problem.add_constraint(AngleConstraint('im1', 'im2', 'im3', np.radians(10)))
problem.add_constraint(AngleConstraint('im2', 'im3', 'im4', np.radians(-10)))
problem.add_constraint(DistanceConstraint('im3', 'im4', 90))
problem.add_constraint(AngleConstraint('im3', 'im4', 'prmbk', np.radians(74)))
problem.add_constraint(DistanceConstraint('im4', 'prmbk', 30))
problem.add_constraint(DistanceConstraint('prmbk', 'prmfr', prm_light_travel_path))
problem.add_constraint(AngleConstraint('im4', 'prmbk', 'prmfr', np.radians(180)))
problem.add_constraint(DistanceConstraint('prmfr', 'pr2', 200))
problem.add_constraint(AngleConstraint('prmbk', 'prmfr', 'pr2', np.radians(180)))
problem.add_constraint(DistanceConstraint('pr2', 'pr3', 160))
problem.add_constraint(AngleConstraint('prmfr', 'pr2', 'pr3', np.radians(15)))
problem.add_constraint(DistanceConstraint('pr3', 'bsfr', 250))
problem.add_constraint(AngleConstraint('pr2', 'pr3', 'bsfr', np.radians(-15)))

problem.add_constraint(DistanceConstraint('bsfr', 'bsbk', bs_light_travel_path))
problem.add_constraint(AngleConstraint('pr3', 'bsfr', 'bsbk', np.radians(bs_aoi)))
problem.add_constraint(DistanceConstraint('bsfr', 'itmy', 75))
problem.add_constraint(AngleConstraint('pr3', 'bsfr', 'itmy', np.radians(90)))
problem.add_constraint(DistanceConstraint('itmy', 'etmy', l_arm))
problem.add_constraint(AngleConstraint('bsfr', 'itmy', 'etmy', np.radians(180)))

# fix the x arm coordinates
problem.add_constraint(FixConstraint('bsbk', Vector.origin()))
problem.add_constraint(FixConstraint('itmx', Vector([75, 0])))
#problem.add_constraint(DistanceConstraint('bsbk', 'itmx', 75))
problem.add_constraint(AngleConstraint('bsfr', 'bsbk', 'itmx', np.radians(-bs_aoi)))
problem.add_constraint(DistanceConstraint('itmx', 'etmx', l_arm))
problem.add_constraint(AngleConstraint('bsbk', 'itmx', 'etmx', np.radians(180)))

problem.add_constraint(DistanceConstraint('bsbk', 'sr3', 75))
problem.add_constraint(AngleConstraint('itmx', 'bsbk', 'sr3', np.radians(100)))
problem.add_constraint(DistanceConstraint('sr3', 'sr2', 50))
problem.add_constraint(AngleConstraint('bsbk', 'sr3', 'sr2', np.radians(-10)))
problem.add_constraint(DistanceConstraint('sr2', 'srm', 60))
problem.add_constraint(AngleConstraint('sr3', 'sr2', 'srm', np.radians(-10)))
problem.add_constraint(DistanceConstraint('srm', 'om1', 70))
problem.add_constraint(AngleConstraint('sr2', 'srm', 'om1', np.radians(180)))
problem.add_constraint(DistanceConstraint('om1', 'om2', 40))
problem.add_constraint(AngleConstraint('srm', 'om1', 'om2', np.radians(-15)))
problem.add_constraint(DistanceConstraint('om2', 'om3', 30))
problem.add_constraint(AngleConstraint('om1', 'om2', 'om3', np.radians(25)))
problem.add_constraint(DistanceConstraint('om3', 'omc1', 10))
problem.add_constraint(AngleConstraint('om2', 'om3', 'omc1', np.radians(-90)))
problem.add_constraint(DistanceConstraint('omc1', 'omc2', 5))
problem.add_constraint(AngleConstraint('om3', 'omc1', 'omc2', np.radians(-90)))
problem.add_constraint(AngleConstraint('omc1', 'omc2', 'omc3', np.radians(180)))
problem.add_constraint(DistanceConstraint('omc2', 'omc3', 30))
problem.add_constraint(AngleConstraint('omc2', 'omc3', 'omc4', np.radians(30)))
problem.add_constraint(AngleConstraint('omc3', 'omc2', 'omc4', np.radians(-90)))
problem.add_constraint(AngleConstraint('omc4', 'omc5', 'omc2', np.radians(-30)))
problem.add_constraint(AngleConstraint('omc3', 'omc4', 'omc5', np.radians(-30)))

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
#nx.drawing.nx_pydot.write_dot(solver.solver._mg, "test_4.dot")
