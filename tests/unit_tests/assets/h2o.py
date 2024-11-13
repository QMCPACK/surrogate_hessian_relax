#!/usr/bin/env python3


from numpy import array, sin, cos, pi
from numpy.random import randn
from .helper import harmonic_a, morse, mean_distances, bond_angle

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# test H2O molecule
pos_H2O = array('''
0.00000        0.00000        0.11779
0.00000        0.75545       -0.47116
0.00000       -0.75545       -0.47116
'''.split(), dtype=float).reshape(-1, 3)
elem_H2O = 'O H H'.split()


def forward_H2O(pos):
    r_OH = mean_distances([(pos[0], pos[1]), (pos[0], pos[2])])
    a_HOH = bond_angle(pos[1], pos[0], pos[2])
    return array([r_OH, a_HOH])
# end def


def backward_H2O(params):
    # r_OH = params[0]
    a_HOH = params[1] * pi / 180
    O1 = [0., 0., 0.]
    H1 = params[0] * array([0.0, cos((pi - a_HOH) / 2), sin((pi - a_HOH) / 2)])
    H2 = params[0] * array([0.0, -cos((pi - a_HOH) / 2), sin((pi - a_HOH) / 2)])
    return array([O1, H1, H2])


# end def
hessian_H2O = array([[1.0, 0.2],
                     [0.2, 0.5]])  # random guess for testing purposes
hessian_real_H2O = array('''
1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 2.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0
0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 4.0 0.0 0.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0
0.0 2.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0
0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 3.0
0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0
0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.2
'''.split(), dtype=float).reshape(9, 9)


def pes_H2O(structure, sigma=0.0):
    r, a = tuple(structure.params)
    V = sigma * randn(1)[0]
    V += morse([0.95789707, 0.5, 0.5, 0.0], r)
    V += harmonic_a([104.119, 0.5], a)
    return V, sigma
# end def


def get_structure_H2O():
    from shapls import ParameterStructure
    return ParameterStructure(forward=forward_H2O, backward=backward_H2O, pos=pos_H2O, elem=elem_H2O)
# end def


def get_hessian_H2O():
    from shapls import ParameterHessian
    return ParameterHessian(hessian=hessian_H2O)
# end def


def job_H2O_pes(structure, path, sigma, **kwargs):
    value = pes_H2O(structure)
    return [(path, value, sigma)]
# end def


def analyze_H2O_pes(path, job_data=None, **kwargs):
    for row in job_data:
        if path == row[0]:
            return row[1]
        # end if
    # end for
    return None
# end def


def get_surrogate_H2O():
    from shapls import TargetParallelLineSearch
    srg = TargetParallelLineSearch(
        structure=get_structure_H2O(),
        hessian=get_hessian_H2O(),
        mode='pes',
        pes_func=pes_H2O,
        M=25,
        window_frac=0.5)
    return srg
# end def
