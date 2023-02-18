#!/usr/bin/env python

from numpy import array
from pytest import raises
from surrogate_classes import match_to_tol

from assets import get_structure_H2O, get_hessian_H2O

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

def test_linesearchbase_class():
    from surrogate_classes import LineSearchBase
    ls = LineSearchBase()  # test generation
    with raises(AssertionError):
        ls.get_x0()
    #end with
    with raises(AssertionError):
        ls.get_y0()
    #end with
    with raises(AssertionError):
        ls.search()
    #end with

    ls = LineSearchBase(
        grid = [0, 1, 2, 3, 4],
        values = [ 2, 1, 0, 1, 2],        
        fit_kind = 'pf3')
    x0 = ls.get_x0()
    y0 = ls.get_y0()
    x0_ref = [2.0000000000, 0.0]
    y0_ref = [0.3428571428, 0.0]
    fit_ref = [ 0.0,  4.28571429e-01, -1.71428571e+00,  2.05714286e+00]
    assert match_to_tol(x0, x0_ref)
    assert match_to_tol(y0, y0_ref)
    assert match_to_tol(ls.fit, fit_ref)

    # test setting wrong number of values
    with raises(AssertionError):
        ls.set_values(values = ls.values[:-2])
    #end with

    # test _search method
    x2, y2, pf2 = ls._search(2 * ls.grid, 2 * ls.values, None)
    assert match_to_tol(x2, 2 * x0_ref[0])
    assert match_to_tol(y2, 2 * y0_ref[0])
    x3, y3, pf3 = ls._search(ls.grid, ls.values, fit_kind = 'pf2')
    assert match_to_tol(x3, [2.0])
    assert match_to_tol(y3, [0.34285714285])
    assert match_to_tol(pf3, [ 0.42857143, -1.71428571,  2.05714286])
    # TODO: more tests
#end def


# test LineSearch
def test_linesearch_class():
    from surrogate_classes import LineSearch
    results = []
    s = get_structure_H2O()
    h = get_hessian_H2O()

    with raises(AssertionError):
        ls_d0 = LineSearch(s, h, d = 1)
    #end with

    ls_d0 = LineSearch(s, h, d = 0, R = 1.3)
    ls_d1 = LineSearch(s, h, d = 1, W = 3.0)
    # test grid generation
    gridR_d0 = ls_d0._make_grid_R(1.3, M = 9)
    gridW_d0 = ls_d0._make_grid_W(3.0, M = 7)
    gridR_d1 = ls_d1._make_grid_R(1.3, M = 7)
    gridW_d1 = ls_d1._make_grid_W(3.0, M = 9)
    gridR_d0_ref = array('-1.3   -0.975 -0.65  -0.325  0.     0.325  0.65   0.975  1.3'.split(),dtype=float)
    gridW_d0_ref = array('-2.36783828 -1.57855885 -0.78927943  0.          0.78927943  1.57855885 2.36783828'.split(),dtype=float)
    gridR_d1_ref = array('-1.3        -0.86666667 -0.43333333  0.          0.43333333  0.86666667  1.3'.split(),dtype=float)
    gridW_d1_ref = array('-3.73611553 -2.80208665 -1.86805777 -0.93402888  0.          0.93402888  1.86805777  2.80208665  3.73611553'.split(),dtype=float)
    assert match_to_tol(gridR_d0, gridR_d0_ref)
    assert match_to_tol(gridR_d1, gridR_d1_ref)
    assert match_to_tol(gridW_d0, gridW_d0_ref)
    assert match_to_tol(gridW_d1, gridW_d1_ref)

    with raises(AssertionError):
        grid, M = ls_d0.figure_out_grid()
    #end with
    with raises(AssertionError):
        grid, M = ls_d0.figure_out_grid(R = -1.0)
    #end with
    with raises(AssertionError):
        grid = ls_d0.figure_out_grid(W = -1.0)
    #end with
#end def

