#!/usr/bin/env python

from numpy import array
from pytest import raises
from shapls.params import PesFunction
from shapls.io import NexusFunction
from shapls.util import match_to_tol

from ..assets.h2o import hessian_H2O, pes_H2O, get_structure_H2O, get_hessian_H2O, job_H2O_pes

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


def test_parallellinesearch_class():
    from shapls import ParallelLineSearch, ParameterSet

    # nexus mode
    pls = ParallelLineSearch(mode='nexus', pes=NexusFunction(job_H2O_pes))
    assert pls.get_status() == '000000'
    s = get_structure_H2O()
    s.shift_params([0.2, 0.2])
    pls.set_structure(s)
    assert pls.get_status() == '000000'
    h = get_hessian_H2O()
    pls.set_hessian(h)
    assert pls.get_status() == '100000'
    pls.guess_windows(windows=None, window_frac=0.1)
    pls.M = 9
    pls.reset_ls_list()

    ls0 = pls.ls_list[0]
    ls1 = pls.ls_list[1]

    # test grid
    ls0_grid_ref = array('''-0.4396967  -0.32977252 -0.21984835 -0.10992417  0.          0.10992417
    0.21984835  0.32977252  0.4396967 '''.split(), dtype=float)
    ls1_grid_ref = array('''-0.55231563 -0.41423672 -0.27615782 -0.13807891  0.          0.13807891
    0.27615782  0.41423672  0.55231563'''.split(), dtype=float)
    assert match_to_tol(ls0.grid, ls0_grid_ref)
    assert match_to_tol(ls1.grid, ls1_grid_ref)
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
    '''.split(), dtype=float)
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
    assert match_to_tol(params0, params0_ref)
    assert match_to_tol(params1, params1_ref)
    # test PES
    values0 = [pes_H2O(ParameterSet(params))[0] for params in params0]
    values1 = [pes_H2O(ParameterSet(params))[0] for params in params1]
    values0_ref = array('''-0.35429145 -0.4647814  -0.49167476 -0.47112498 -0.42546898 -0.36820753
   -0.30724027 -0.24695829 -0.18959033'''.split(), dtype=float)
    values1_ref = array('''-0.3056267  -0.3616872  -0.40068042 -0.42214136 -0.42546898 -0.40989479
   -0.37444477 -0.31789329 -0.23870716'''.split(), dtype=float)
    assert match_to_tol(values0, values0_ref)
    assert match_to_tol(values1, values1_ref)

    pls.status.generated = True
    with raises(AssertionError):
        pls.load_results()  # loading with empty data does not work
    # manually enter values
    pls.load_results(values=[values0, values1])
    assert pls.get_status() == '111110'
    # propagate
    # pls_next = pls.propagate()
    # assert pls.get_status() == '111111'
    # assert pls_next.get_status() == '110000'

    # ls0_x0_ref = -0.19600534, 0.0
    # ls0_y0_ref = -0.48854587, 0.0
    # ls1_x0_ref = -0.04318508, 0.0
    # ls1_y0_ref = -0.42666697, 0.0
    # assert match_to_tol(ls0.get_x0(), ls0_x0_ref)
    # assert match_to_tol(ls0.get_y0(), ls0_y0_ref)
    # assert match_to_tol(ls1.get_x0(), ls1_x0_ref)
    # assert match_to_tol(ls1.get_y0(), ls1_y0_ref)

    next_params_ref = [0.98723545, 104.21430094]
    assert match_to_tol(pls.get_next_params(), next_params_ref)

    # test init from hessian array, also switch units, do pes mode
    pls = ParallelLineSearch(
        structure=s,
        hessian=hessian_H2O,
        pes=PesFunction(pes_H2O),
        M=5,
        x_unit='B',
        E_unit='Ha',
        mode='pes',
        window_frac=0.1,
        noises=None,)
    assert match_to_tol(pls.Lambdas, [0.074919, 0.030092], tol=1e-5)

    # test partial line-search
    pls.reset_ls_list(D=[1])
    pls.load_results()
    assert match_to_tol(pls.propagate().structure.params,
                        [1.170805, 104.283132], tol=1e-5)

    from shutil import rmtree
    rmtree('pls/')
# end def
