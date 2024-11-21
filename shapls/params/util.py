from numpy import linalg, pi, arccos, array, dot, loadtxt
from scipy.optimize import minimize


def distance(r0, r1):
    '''Return Euclidean distance between two positions'''
    r = linalg.norm(r0 - r1)
    return r
# end def


def bond_angle(r0, rc, r1, units='ang'):
    '''Return dihedral angle between 3 bodies'''
    v1 = r0 - rc
    v2 = r1 - rc
    cosang = dot(v1, v2) / linalg.norm(v1) / linalg.norm(v2)
    ang = arccos(cosang) * 180 / pi if units == 'ang' else arccos(cosang)
    return ang
# end def


def mean_distances(pairs):
    '''Return average distance over (presumably) identical position pairs'''
    rs = []
    for pair in pairs:
        rs.append(distance(pair[0], pair[1]))
    # end for
    return array(rs).mean()
# end def


def mean_param(params, tol=1e-6):
    avg = array([params]).mean()
    if not all(params - avg < tol):
        print("Warning! Some of symmetric parameters stand out:")
        print(params)
    # end if
    return avg
# end def


def invert_pos(pos0, params, forward=None, tol=1.0e-7, method='BFGS'):
    assert forward is not None, 'Must provide forward mapping'

    def dparams_sum(pos1):
        return sum((params - forward(pos1))**2)
    # end def
    pos1 = minimize(dparams_sum, pos0, tol=tol, method=method).x
    return pos1
# end def
