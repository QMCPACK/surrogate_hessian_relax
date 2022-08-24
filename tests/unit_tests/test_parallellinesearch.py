#!/usr/bin/env python

from numpy import array, exp, nan, isnan, random, polyval, linspace
from pytest import raises
from testing import match_values, add_unit_test

from unit_tests.assets import pos_H2O, elem_H2O, forward_H2O, backward_H2O, hessian_H2O, pes_H2O, hessian_real_H2O, get_structure_H2O, get_hessian_H2O
from unit_tests.assets import pos_H2, elem_H2, forward_H2, backward_H2, hessian_H2, get_structure_H2, get_hessian_H2, get_surrogate_H2O
from unit_tests.assets import params_GeSe, forward_GeSe, backward_GeSe, hessian_GeSe, elem_GeSe
from unit_tests.assets import morse, Gs_N200_M7


def test_parallellinesearch_class():
    from surrogate_classes import ParallelLineSearch
    s = get_structure_H2O()
    s.shift_params([0.2, 0.2])
    h = get_hessian_H2O()
    pls = ParallelLineSearch(
        hessian = h,
        structure = s,
        M = 9,
        window_frac = 0.1,
        noise_frac = 0.0)
    assert not pls.protected
    assert not pls.generated
    assert not pls.loaded
    assert not pls.calculated

    ls0 = pls.ls_list[0]
    ls1 = pls.ls_list[1]

    # test grid
    ls0_grid_ref = array('''-0.4396967  -0.32977252 -0.21984835 -0.10992417  0.          0.10992417
  0.21984835  0.32977252  0.4396967 '''.split(), dtype=float)
    ls1_grid_ref = array('''-0.55231563 -0.41423672 -0.27615782 -0.13807891  0.          0.13807891
  0.27615782  0.41423672  0.55231563'''.split(), dtype=float)
    assert match_values(ls0.grid, ls0_grid_ref)
    assert match_values(ls1.grid, ls1_grid_ref)
    # test params
    params0 = pls.get_shifted_params(0)
    params1 = pls.get_shifted_params(1)
    params0_ref = array('''
    0.74298682 104.17438807
    0.84671438 104.21077373
    0.95044195 104.2471594 
    1.05416951 104.28354506
    1.15789707 104.31993072
    1.26162464 104.35631639
    1.3653522  104.39270205
    1.46907977 104.42908772
    1.57280733 104.46547338
    '''.split(),dtype=float)
    params1_ref = array('''
    1.34071738 103.79875005
    1.29501231 103.92904522
    1.24930723 104.05934039
    1.20360215 104.18963556
    1.15789707 104.31993072
    1.112192   104.45022589
    1.06648692 104.58052106
    1.02078184 104.71081623
    0.97507677 104.84111139
    '''.split(), dtype=float)
    assert match_values(params0, params0_ref)
    assert match_values(params1, params1_ref)
    # test PES
    values0 = [pes_H2O(params) for params in params0]
    values1 = [pes_H2O(params) for params in params1]
    values0_ref = array('''-0.35429145 -0.4647814  -0.49167476 -0.47112498 -0.42546898 -0.36820753
 -0.30724027 -0.24695829 -0.18959033'''.split(), dtype = float)
    values1_ref = array('''-0.3056267  -0.3616872  -0.40068042 -0.42214136 -0.42546898 -0.40989479
 -0.37444477 -0.31789329 -0.23870716'''.split(), dtype = float)
    assert match_values(values0, values0_ref)
    assert match_values(values1, values1_ref)

    # test loading without function
    pls.load_results()
    assert not pls.loaded
    # manually enter values
    pls.load_results(values=[values0, values1])
    assert pls.loaded

    ls0_x0_ref = -0.19600534, 0.0
    ls0_y0_ref = -0.48854587, 0.0
    ls1_x0_ref = -0.04318508, 0.0
    ls1_y0_ref = -0.42666697, 0.0
    assert match_values(ls0.get_x0(), ls0_x0_ref)
    assert match_values(ls0.get_y0(), ls0_y0_ref)
    assert match_values(ls1.get_x0(), ls1_x0_ref)
    assert match_values(ls1.get_y0(), ls1_y0_ref)
  
    next_params_ref = [0.98723545, 104.21430094]
    assert match_values(pls.get_next_params(), next_params_ref)

    # test init from hessian array, also switch units
    pls = ParallelLineSearch(
        structure = s,
        hessian = hessian_H2O,
        M = 5,
        x_unit = 'B',
        E_unit = 'Ha',
        window_frac = 0.1,
        noise_frac = 0.0)
    assert match_values(pls.Lambdas, [0.074919, 0.030092], tol = 1e-5)
    # TODO:
#end def
add_unit_test(test_parallellinesearch_class)

