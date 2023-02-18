#!/usr/bin/env python

from numpy import array, exp, nan, isnan, random, polyval
from pytest import raises
from surrogate_classes import match_to_tol

from assets import hessian_H2O, get_structure_H2O, hessian_real_H2O

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

def test_parameterhessian_class():
    from surrogate_classes import ParameterHessian
    # init from parameter hessian array
    h = ParameterHessian(hessian_H2O)
    Lambda = h.Lambda
    dire = h.get_directions()
    dir0 = h.get_directions(0)
    dir1 = h.get_directions(1)
    Lambda_ref = array([1.07015621, 0.42984379])
    dire_ref = array([[ 0.94362832,  0.33100694],
                      [-0.33100694,  0.94362832]])
    with raises(IndexError):
        dir2 = h.get_directions(2)
    #end with
    assert match_to_tol(dire, dire_ref)
    assert match_to_tol(dir0, dire_ref[0])
    assert match_to_tol(dir1, dire_ref[1])
    assert match_to_tol(Lambda, Lambda_ref)
    # test update hessian
    h.update_hessian(hessian_H2O**2)
    assert match_to_tol(h.get_directions(0), [ 0.9985888, 0.05310744], tol = 1e-5)
    assert match_to_tol(h.get_directions(1), [-0.05310744, 0.9985888], tol = 1e-5)
    assert match_to_tol(h.Lambda, [1.002127, 0.247873], tol = 1e-5)

    # init hessian from structure and real-space hessian
    s = get_structure_H2O()
    h = ParameterHessian(structure = s, hessian_real = hessian_real_H2O)
    h_ref = array('''
    3.008569 0.005269 
    0.005269 0.000188 
    '''.split(), dtype = float)
    assert match_to_tol(h.hessian, h_ref, tol = 1e-5)
#end def

