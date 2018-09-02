import numpy as np
import random

from .geometry import Vector
from .matfunc import Mat,Vec,eye
from .tolerance import *
from .diagnostic import *

# ------ misc fucntions ----------

def sign(x):
    """Returns 1 if x>0, return -1 if x<0 and 0 if x=0"""
    if tol_gt(x,0.0):
        return 1.0
    elif tol_lt(x,0.0):
        return -1.0
    else:
        return 0.0

def sign2(x):
    """Returns 1 if x>0, else -1 (even if x=0)"""
    if tol_gt(x,0.0):
        return 1.0
    else:
        return -1.0

# ------- 2D intersections ----------------

def cc_int(p1, r1, p2, r2):
    """
    Intersect circle (p1,r1) circle (p2,r2)
    where p1 and p2 are 2-vectors and r1 and r2 are scalars
    Returns a list of zero, one or two solution points.
    """
    d = p2.distance_to(p1)
    if not tol_gt(d, 0):
        return []
    u = ((r1*r1 - r2*r2)/d + d)/2
    if tol_lt(r1*r1, u*u):
        return []
    elif r1*r1 < u*u:
        v = 0.0
    else:
        v = np.sqrt(r1 * r1 - u * u)
    s = (p2-p1) * u / d
    if tol_eq(s.length, 0):
        p3a = p1+Vector([p2[1]-p1[1],p1[0]-p2[0]]) * r1 / d
        if tol_eq(r1/d,0):
            return [p3a]
        else:
            p3b = p1+Vector([p1[1]-p2[1],p2[0]-p1[0]])*r1/d
            return [p3a,p3b]
    else:
        p3a = p1 + s + Vector([s[1], -s[0]]) * v / s.length
        if tol_eq(v / s.length, 0):
            return [p3a]
        else:
            p3b = p1 + s + Vector([-s[1], s[0]]) * v / s.length
            return [p3a,p3b]


def cl_int(p1,r,p2,v):
    """
    Intersect a circle (p1,r) with line (p2,v)
    where p1, p2 and v are 2-vectors, r is a scalar
    Returns a list of zero, one or two solution points
    """
    p = p2 - p1
    d2 = v[0]*v[0] + v[1]*v[1]
    D = p[0]*v[1] - v[0]*p[1]
    E = r*r*d2 - D*D
    if tol_gt(d2, 0) and tol_gt(E, 0):
        sE = np.sqrt(E)
        x1 = p1[0] + (D * v[1] + sign2(v[1])*v[0]*sE) / d2
        x2 = p1[0] + (D * v[1] - sign2(v[1])*v[0]*sE) / d2
        y1 = p1[1] + (-D * v[0] + abs(v[1])*sE) / d2
        y2 = p1[1] + (-D * v[0] - abs(v[1])*sE) / d2
        return [Vector([x1,y1]), Vector([x2,y2])]
    elif tol_eq(E, 0):
        x1 = p1[0] + D * v[1] / d2
        y1 = p1[1] + -D * v[0] / d2
        return [Vector([x1,y1])]
    else:
        return []

def cr_int(p1,r,p2,v):
    """
    Intersect a circle (p1,r) with ray (p2,v) (a half-line)
    where p1, p2 and v are 2-vectors, r is a scalar
    Returns a list of zero, one or two solutions.
    """
    sols = []
    all = cl_int(p1,r,p2,v)
    for s in all:
        if tol_gte(np.dot(s-p2, v), 0):
            sols.append(s)
    return sols

def ll_int(p1, v1, p2, v2):
    """Intersect line though p1 direction v1 with line through p2 direction v2.
       Returns a list of zero or one solutions
    """
    diag_print("ll_int "+str(p1)+str(v1)+str(p2)+str(v2),"intersections")
    if tol_eq((v1[0]*v2[1])-(v1[1]*v2[0]),0):
        return []
    elif not tol_eq(v2[1],0.0):
        d = p2-p1
        r2 = -v2[0]/v2[1]
        f = v1[0] + v1[1]*r2
        t1 = (d[0] + d[1]*r2) / f
    else:
        d = p2-p1
        t1 = d[1]/v1[1]
    return [p1 + v1*t1]

def lr_int(p1, v1, p2, v2):
    """Intersect line though p1 direction v1 with ray through p2 direction v2.
       Returns a list of zero or one solutions
    """
    diag_print("lr_int "+str(p1)+str(v1)+str(p2)+str(v2),"intersections")
    s = ll_int(p1,v1,p2,v2)
    if len(s) > 0 and tol_gte(np.dot(s[0] - p2, v2), 0):
        return s
    else:
        return []

def rr_int(p1, v1, p2, v2):
    """Intersect ray though p1 direction v1 with ray through p2 direction v2.
       Returns a list of zero or one solutions
    """
    diag_print("rr_int "+str(p1)+str(v1)+str(p2)+str(v2),"intersections")
    s = ll_int(p1,v1,p2,v2)
    if len(s) > 0 and tol_gte(np.dot(s[0] - p2, v2), 0) and tol_gte(np.dot(s[0] - p1, v1), 0):
        return s
    else:
        return []

# ----- Geometric properties -------

def angle_3p(p1, p2, p3):
    """Returns the angle, in radians, rotating vector p2p1 to vector p2p3.
       arg keywords:
          p1 - a vector
          p2 - a vector
          p3 - a vector
       returns: a number
       In 2D, the angle is a signed angle, range [-pi,pi], corresponding
       to a clockwise rotation. If p1-p2-p3 is clockwise, then angle > 0.
       In 3D, the angle is unsigned, range [0,pi]
    """
    d21 = p2.distance_to(p1)
    d23 = p3.distance_to(p2)
    if tol_eq(d21,0) or tol_eq(d23,0):
        # degenerate, indeterminate angle
        return None
    v21 = (p1 - p2) / d21
    v23 = (p3 - p2) / d23
    t = np.dot(v21, v23) # / (d21 * d23)
    if t > 1.0:             # check for floating point error
        t = 1.0
    elif t < -1.0:
        t = -1.0
    angle = np.arccos(t)
    if len(p1) == 2:        # 2D case
        if is_counterclockwise(p1,p2,p3):
            angle = -angle
    return angle

def distance_2p(p1, p2):
    """Returns the euclidean distance between two points
       arg keywords:
          p1 - a vector
          p2 - a vector
       returns: a number
    """
    return p2.distance_to(p1)

def distance_point_line(p,l1,l2):
    """distance from point p to line l1-l2"""
    # v,w is p, l2 relative to l1
    v = p - l1
    w = l2 - l1
    # x = projection v on w
    if tol_eq(w.length, 0):
        x = 0*w
    else:
        x = w * np.dot(v,w) / w.length
    # result is distance x,v
    return x.distance_to(v)

# ------ 2D

def is_clockwise(p1,p2,p3):
    """ returns True iff triangle p1,p2,p3 is clockwise oriented"""
    assert len(p1)==2
    assert len(p1)==len(p2)
    assert len(p2)==len(p3)
    u = p2 - p1
    v = p3 - p2
    perp_u = Vector([-u[1], u[0]])
    return tol_lt(np.dot(perp_u,v),0)

def is_counterclockwise(p1,p2,p3):
    """ returns True iff triangle p1,p2,p3 is counterclockwise oriented"""
    assert len(p1)==2
    assert len(p1)==len(p2)
    assert len(p2)==len(p3)
    u = p2 - p1
    v = p3 - p2;
    perp_u = Vector([-u[1], u[0]])
    return tol_gt(np.dot(perp_u,v), 0)

def is_colinear(p1,p2,p3):
    """ returns True iff triangle p1,p2,p3 is colinear (neither clockwise of counterclockwise oriented)"""
    assert len(p1)==2
    assert len(p1)==len(p2)
    assert len(p2)==len(p3)

    u = p2 - p1
    v = p3 - p2;
    perp_u = Vector([-u[1], u[0]])
    return tol_eq(np.dot(perp_u,v), 0)

def is_acute(p1,p2,p3):
    """returns True iff angle p1,p2,p3 is acute, i.e. less than pi/2"""
    angle = angle_3p(p1, p2, p3)
    if angle == None:
        return None
    else:
        return tol_gt(abs(angle), np.pi / 2)


def is_obtuse(p1,p2,p3):
    """returns True iff angle p1,p2,p3 is obtuse, i.e. greater than pi/2"""
    angle = angle_3p(p1, p2, p3)
    if angle == None:
        return None
    else:
        return tol_gt(abs(angle), np.pi / 2)

# --------- coordinate tranformations -------

def make_hcs_2d (a, b):
    """build a 2D homogeneus coordiate system from two points"""
    u = b-a
    if tol_eq(u.length, 0.0):
        u = Vector([0.0,0.0])
    else:
        u = u.unit()
    v = Vector([-u[1], u[0]])
    hcs = Mat([ [u[0],v[0],a[0]] , [u[1],v[1],a[1]] , [0.0, 0.0, 1.0] ] )
    return hcs

def make_hcs_2d_scaled (a, b):
    """build a 2D homogeneus coordiate system from two points, but scale with distance between input point"""
    u = b-a
    if tol_eq(u.length, 0.0):
        u = Vector([1.0,0.0])

    v = Vector([-u[1], u[0]])
    hcs = Mat([ [u[0],v[0],a[0]] , [u[1],v[1],a[1]] , [0.0, 0.0, 1.0] ] )
    return hcs

def cs_transform_matrix(from_cs, to_cs):
    """returns a transform matrix from from_cs to to_cs"""
    try:
        transform = to_cs.mmul(from_cs.inverse())
    except Exception as e:
        print("from_cs=",from_cs)
        raise Exception("from_cs is not a valid coodinate system.")
    return transform

# ------- rigid transformations ----------

#--- 2D

def id_transform_2D():
    return eye(3)

def translate_2D(dx,dy):
    mat = Mat([
        [1.0, 0.0, dx] ,
        [0.0, 1.0, dy] ,
        [0.0, 0.0, 1.0] ] )
    return mat

def rotate_2D(angle):
    """rotation matrix for rotation in 2d with homogeneous coordinates"""
    cosa = np.cos(angle)
    sina = np.sin(angle)
    mat = Mat([
        [cosa,-sina,0.0],
        [sina,cosa,0.0],
        [0.0, 0.0, 1.0] ])
    return mat

# ---- applyign transformations

def transform_point(point, transform):
    """transform a point"""
    hpoint = Vec(point)
    hpoint.append(1.0)
    hres = transform.mmul(hpoint)
    res = Vector(hres[0:-1]) / hres[-1]
    return res

# -------- perpendicular vectors ---------

def perp_2d(v):
    """returns the orthonormal vector."""
    return Vector([-v[1], v[0]]).unit()

# -------------------------test code -----------------

def test_ll_int():
    """test random line-line intersection. returns True iff succesful"""
    # generate tree points A,B,C an two lines AC, BC.
    # then calculate the intersection of the two lines
    # and check that it equals C
    p_a = vector.randvec(2, 0.0, 10.0,1.0)
    p_b = vector.randvec(2, 0.0, 10.0,1.0)
    p_c = vector.randvec(2, 0.0, 10.0,1.0)
    # print p_a, p_b, p_c
    if tol_eq(p_c.distance_to(p_a), 0) or tol_eq(p_c.distance_to(p_b), 0):
        return True # ignore this case
    v_ac = (p_c - p_a) / p_c.distance_to(p_a)
    v_bc = (p_c - p_b) / p_c.distance_to(p_b)
    s = ll_int(p_a, v_ac, p_b, v_bc)
    if tol_eq(np.absolute(np.dot(v_ac, v_bc)), 1.0):
        return len(s) == 0
    else:
        if len(s) > 0:
            p_s = s[0]
            return tol_eq(p_s[0],p_c[0]) and tol_eq(p_s[1],p_c[1])
        else:
            return False

def test_rr_int():
    """test random ray-ray intersection. returns True iff succesful"""
    # generate tree points A,B,C an two rays AC, BC.
    # then calculate the intersection of the two rays
    # and check that it equals C
    p_a = vector.randvec(2, 0.0, 10.0,1.0)
    p_b = vector.randvec(2, 0.0, 10.0,1.0)
    p_c = vector.randvec(2, 0.0, 10.0,1.0)
    # print p_a, p_b, p_c
    if tol_eq(p_c.distance_to(p_a), 0) or tol_eq(p_c.distance_to(p_b), 0):
        return True # ignore this case
    v_ac = (p_c - p_a).unit()
    v_bc = (p_c - p_b).unit()
    s = rr_int(p_a, v_ac, p_b, v_bc)
    if tol_eq(np.absolute(np.dot(v_ac, v_bc)), 1.0):
        return len(s) == 0
    else:
        if len(s) > 0:
            p_s = s[0]
            return tol_eq(p_s[0],p_c[0]) and tol_eq(p_s[1],p_c[1])
        else:
            return False

def test_sss_int():
    p1 = vector.randvec(3, 0.0, 10.0,1.0)
    p2 = vector.randvec(3, 0.0, 10.0,1.0)
    p3 = vector.randvec(3, 0.0, 10.0,1.0)
    p4 = vector.randvec(3, 0.0, 10.0,1.0)
    #p1 = Vector([0.0,0.0,0.0])
    #p2 = Vector([1.0,0.0,0.0])
    #p3 = Vector([0.0,1.0,0.0])
    #p4 = Vector([1.0,1.0,1.0])
    d14 = p4.distance_to(p1)
    d24 = p4.distance_to(p2)
    d34 = p4.distance_to(p3)
    sols = sss_int(p1,d14,p2,d24,p3,d34)
    sat = True
    for sol in sols:
        # print sol
        d1s = sol.distance_to(p1)
        d2s = sol.distance_to(p2)
        d3s = sol.distance_to(p3)
        sat = sat and tol_eq(d1s,d14)
        sat = sat and tol_eq(d2s,d24)
        sat = sat and tol_eq(d3s,d34)
        # print sat
    return sat

def test_cc_int():
    """Generates two random circles, computes the intersection,
       and verifies that the number of intersections and the
       positions of the intersections are correct.
       Returns True or False"""
    # gen two random circles p1,r2 and p2, r2
    p1 = vector.randvec(2, 0.0, 10.0,1.0)
    r1 = random.uniform(0, 10.0)
    p2 = vector.randvec(2, 0.0, 10.0,1.0)
    r2 = random.uniform(0, 10.0)
    # 33% chance that r2=abs(r1-|p1-p2|)
    if random.random() < 0.33:
        r2 = abs(r1 - p1.distance_to(p2))
    # do interesection
    diag_print("problem:"+str(p1)+","+str(r1)+","+str(p2)+","+str(r2),"test_cc_int")
    sols = cc_int(p1, r1, p2, r2)
    diag_print("solutions:"+str(list(map(str, sols))),"test_cc_int")
    # test number of intersections
    if len(sols) == 0:
        if not tol_gt(p2.distance_to(p1),r1 + r2) and not tol_lt(p2.distance_to(p1),abs(r1 - r2)) and not tol_eq(p1.distance_to(p2),0):
            diag_print("number of solutions 0 is wrong","test_cc_int")
            return False
    elif len(sols) == 1:
        if not tol_eq(p2.distance_to(p1),r1 + r2) and not tol_eq(p2.distance_to(p1),abs(r1-r2)):
            diag_print("number of solutions 1 is wrong","test_cc_int")
            return False
    elif len(sols) == 2:
        if not tol_lt(p2.distance_to(p1),r1 + r2) and not tol_gt(p2.distance_to(p1),abs(r1 - r2)):
            diag_print("number of solutions 2 is wrong")
            return False
    else:
        diag_print("number of solutions > 2 is wrong","test_cc_int")
        return False

    # test intersection coords
    for p3 in sols:
        if not tol_eq(p3.distance_to(p1), r1):
            diag_print("solution not on circle 1","test_cc_int")
            return False
        if not tol_eq(p3.distance_to(p2), r2):
            diag_print("solution not on circle 2","test_cc_int")
            return False

    diag_print("OK","test_cc_int")
    return True

def test_cl_int():
    """Generates random circle and line, computes the intersection,
       and verifies that the number of intersections and the
       positions of the intersections are correct.
       Returns True or False"""
    # 3 random points
    p1 = vector.randvec(2, 0.0, 10.0,1.0)
    p2 = vector.randvec(2, 0.0, 10.0,1.0)
    p3 = vector.randvec(2, 0.0, 10.0,1.0)
    # prevent div by zero / no line direction
    if tol_eq(p1.distance_to(p2),0):
        p2 = p1 + p3 * 0.1
    # line (o,v): origin p1, direction p1-p2
    o = p1
    v = (p2 - p1).unit()
    # cirlce (c, r): centered in p3, radius p3-p2 + rx
    c = p3
    r0 = p3.distance_to(p2)
    # cases rx = 0, rx > 0, rx < 0
    case = random.choice([1,2,3])
    if case==1:
        r = r0      #should have one intersection (unles r0 = 0)
    elif case==2:
        r = random.random() * r0   # should have no ints (unless r0=0)
    elif case==3:
        r = r0 + random.random() * r0 # should have 2 ints (unless r0=0)
    # do interesection
    diag_print("problem:"+str(c)+","+str(r)+","+str(o)+","+str(v),"test_cl_int")
    sols = cl_int(c,r,o,v)
    diag_print("solutions:"+str(list(map(str, sols))),"test_cl_int")
    # distance from point on line closest to circle center
    l = np.dot(c-o, v) / v.length
    p = o + v * l / v.length
    d = p.distance_to(c)
    diag_print("distance center to line="+str(d),"test_cl_int")
    # test number of intersections
    if len(sols) == 0:
        if not tol_gt(d, r):
            diag_print("wrong number of solutions: 0", "test_cl_int")
            return False
    elif len(sols) == 1:
        if not tol_eq(d, r):
            diag_print("wrong number of solutions: 1", "test_cl_int")
            return False
    elif len(sols) == 2:
        if not tol_lt(d, r):
            diag_print("wrong number of solutions: 2", "test_cl_int")
            return False
    else:
            diag_print("wrong number of solutions: >2", "test_cl_int")

    # test coordinates of intersection
    for s in sols:
        # s on line (o,v)
        if not is_colinear(s, o, o+v):
            diag_print("solution not on line", "test_cl_int")
            return False
        # s on circle c, r
        if not tol_eq(s.distance_to(c), r):
            diag_print("solution not on circle", "test_cl_int")
            return False

    return True


def test_intersections():
    sat = True
    for i in range(0,100):
        sat = sat and test_ll_int()
        if not sat:
            print("ll_int() failed")
            return
    if sat:
        print("ll_int() passed")
    else:
        print("ll_int() FAILED")

    sat = True
    for i in range(0,100):
        sat = sat and test_rr_int()
        if not sat:
            print("rr_int() failed")
            return
    if sat:
        print("rr_int() passed")
    else:
        print("rr_int() FAILED")

    sat = True
    for i in range(0,100):
        sat = sat and test_cc_int()
    if sat:
        print("cc_int() passed")
    else:
        print("cc_int() FAILED")

    sat = True
    for i in range(0,100):
        sat = sat and test_cl_int()
    if sat:
        print("cl_int() passed")
    else:
        print("cl_int() FAILED")

    sat = True
    for i in range(0,100):
        sat = sat and test_sss_int()
    if sat:
        print("sss_int() passed")
    else:
        print("sss_int() FAILED")
    print("Note: did not test degenerate cases for sss_int")

def test_angles():
    print("2D angles")
    for i in range(9):
        a = i * 45 * np.pi / 180
        p1 = Vector([1.0,0.0])
        p2 = Vector([0.0,0.0])
        p3 = Vector([np.cos(a), np.sin(a)])
        print(p3, angle_3p(p1,p2,p3) * 180 / np.pi, "flip", angle_3p(p3,p2,p1) * 180 / np.pi)

    print("3D angles")
    for i in range(9):
        a = i * 45 * np.pi / 180
        p1 = Vector([1.0,0.0,0.0])
        p2 = Vector([0.0,0.0,0.0])
        p3 = Vector([np.cos(a),np.sin(a),0.0])
        print(p3, angle_3p(p1,p2,p3) * 180 / np.pi, "flip", angle_3p(p3,p2,p1) * 180 / np.pi)


def test_perp_3d():
    for i in range(10):
        u = vector.randvec(3).unit()
        v,w = perp_3d(u)
        print(u, np.norm(u))
        print(v, np.norm(v))
        print(w, np.norm(w))
        print(tol_eq(np.dot(u,v),0.0))
        print(tol_eq(np.dot(v,w),0.0))
        print(tol_eq(np.dot(w,u),0.0))

def test_sss_degen():
    p1 = Vector([0.0, 0.0, 0.0])
    p2 = Vector([2.0, 0.0, 0.0])
    p3 = Vector([1.0, 0.0, 0.0])
    r1 = 2.0
    r2 = 2.0
    r3 = np.sqrt(3)
    print(sss_int(p1,r1,p2,r2,p3,r3))

def test_hcs_degen():
    p1 = Vector([0.0, 0.0, 0.0])
    p2 = Vector([1.0, 0.0, 0.0])
    p3 = Vector([1.0, 0.0, 0.0])
    print(make_hcs_3d(p1,p2,p3))

if __name__ == '__main__':
    #diag_select("test_cc_int")
    #diag_select("test_cl_int")
    #diag_select(".*")
    test_intersections()
    #test_angles()
    #test_perp_3d()
    #test_sss_degen()
    #test_hcs_degen()
