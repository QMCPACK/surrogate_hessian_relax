#!/usr/bin/env python

from numpy import array, exp, nan, isnan, random, polyval
from pytest import raises
from testing import match_values, add_unit_test

from unit_tests.assets import pos_H2O, elem_H2O, forward_H2O, backward_H2O, hessian_H2O, pes_H2O, hessian_real_H2O, get_structure_H2O, get_hessian_H2O
from unit_tests.assets import pos_H2, elem_H2, forward_H2, backward_H2, hessian_H2, get_structure_H2, get_hessian_H2, get_surrogate_H2O
from unit_tests.assets import params_GeSe, forward_GeSe, backward_GeSe, hessian_GeSe, elem_GeSe
from unit_tests.assets import morse, Gs_N200_M7



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
    assert match_values(dire, dire_ref)
    assert match_values(dir0, dire_ref[0])
    assert match_values(dir1, dire_ref[1])
    assert match_values(Lambda, Lambda_ref)
    # test update hessian
    h.update_hessian(hessian_H2O**2)
    assert match_values(h.get_directions(0), [ 0.9985888, 0.05310744], tol = 1e-5)
    assert match_values(h.get_directions(1), [-0.05310744, 0.9985888], tol = 1e-5)
    assert match_values(h.Lambda, [1.002127, 0.247873], tol = 1e-5)

    # init hessian from structure and real-space hessian
    s = get_structure_H2O()
    h = ParameterHessian(structure = s, hessian_real = hessian_real_H2O)
    h_ref = array('''
    3.008569 0.005269 
    0.005269 0.000188 
    '''.split(), dtype = float)
    assert match_values(h.hessian, h_ref, tol = 1e-5)
#end def
add_unit_test(test_parameterhessian_class)

