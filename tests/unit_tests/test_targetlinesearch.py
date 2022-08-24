#!/usr/bin/env python

from numpy import array, exp, nan, isnan, random, polyval, linspace
from pytest import raises
from testing import match_values, add_unit_test

from unit_tests.assets import pos_H2O, elem_H2O, forward_H2O, backward_H2O, hessian_H2O, pes_H2O, hessian_real_H2O, get_structure_H2O, get_hessian_H2O
from unit_tests.assets import pos_H2, elem_H2, forward_H2, backward_H2, hessian_H2, get_structure_H2, get_hessian_H2, get_surrogate_H2O
from unit_tests.assets import params_GeSe, forward_GeSe, backward_GeSe, hessian_GeSe, elem_GeSe
from unit_tests.assets import morse, Gs_N200_M7


def test_targetlinesearchbase_class():
    from surrogate_classes import TargetLineSearchBase

    # generate reference potential
    p = [1.0, 0.5, 0.5, 0.0]
    grid = linspace(-0.5, 0.5, 101)
    values = morse(p, grid + 1.0)

    ls = TargetLineSearchBase(
        fit_kind = 'pf3',
        errorbar = 0.025,
        bias_x0 = 0.0,
        bias_y0 = -0.5,
        bias_mix = 0.0,
        target_grid = grid,
        target_values = values,
    )

    bias_x, bias_y, bias_tot = ls.compute_bias(grid, values)
    # TODO
#end def
add_unit_test(test_targetlinesearchbase_class)


# test TargetLineSearch class
def test_targetlinesearch_class():
    from surrogate_classes import TargetLineSearch

    results = []
    # generate reference potential
    p = [1.0, 0.5, 0.5, 0.0]
    grid = linspace(-0.51, 0.51, 21)
    values = morse(p, grid + 1.0)

    s = get_structure_H2O()
    h = get_hessian_H2O()

    # bias_mix = 0
    tls0 = TargetLineSearch(
        structure = s,
        hessian = h,
        d = 0,
        W = 0.5,
        fit_kind = 'pf3',
        M = 9,
        errorbar = 0.025,
        target_x0 = 0.0,
        target_y0 = -0.5,
        bias_mix = 0.0,
    )
    tls0.set_target(grid, values, interpolate_kind = 'pchip')  # should be cubic interpolation, as instructed above
    Rs = linspace(1e-2, 0.5, 21)
    biases_x, biases_y, biases_tot = tls0.compute_bias_of(Rs, M = 7, verbose = False)
    biases_ref = array('''
    0.010000   0.000124   0.000002   0.000124   
    0.034500   0.000519   0.000067   0.000519   
    0.059000   0.000927   0.000283   0.000927   
    0.083500   0.000033   0.000123   0.000033   
    0.108000   0.000226   0.000097   0.000226   
    0.132500   0.000197   -0.000017  0.000197   
    0.157000   -0.000241  -0.000410  0.000241   
    0.181500   -0.000486  -0.000861  0.000486   
    0.206000   -0.000430  -0.001236  0.000430   
    0.230500   -0.000786  -0.001831  0.000786   
    0.255000   -0.001153  -0.002652  0.001153   
    0.279500   -0.001491  -0.003763  0.001491   
    0.304000   -0.002201  -0.005432  0.002201   
    0.328500   -0.003040  -0.007614  0.003040   
    0.353000   -0.003817  -0.010235  0.003817   
    0.377500   -0.004788  -0.013485  0.004788   
    0.402000   -0.006060  -0.017627  0.006060   
    0.426500   -0.007292  -0.022583  0.007292   
    0.451000   -0.008864  -0.028820  0.008864   
    0.475500   -0.010680  -0.036409  0.010680   
    0.500000   -0.012562  -0.045427  0.012562  
    '''.split(),dtype=float).reshape(-1,4)
    assert match_values(Rs, biases_ref[:,0], tol=1e-5)
    assert match_values(biases_x, biases_ref[:,1], tol=1e-5)
    assert match_values(biases_y, biases_ref[:,2], tol=1e-5)
    assert match_values(biases_tot, biases_ref[:,3], tol=1e-5)

    # bias_mix = 0.4, pf4, cubic
    tls0.set_target(grid, values, interpolate_kind = 'cubic')
    biases_x, biases_y, biases_tot = tls0.compute_bias_of(R=Rs, fit_kind = 'pf4', bias_mix = 0.4, M = 7, verbose = False)
    biases_ref = array('''
    0.010000   -0.000005  -0.000000  0.000005   
    0.034500   -0.000005  -0.000000  0.000005   
    0.059000   -0.000005  -0.000001  0.000005   
    0.083500   -0.000016  -0.000001  0.000016   
    0.108000   -0.000044  -0.000000  0.000045   
    0.132500   -0.000096  0.000000   0.000096   
    0.157000   -0.000190  0.000001   0.000190   
    0.181500   -0.000342  0.000002   0.000343   
    0.206000   -0.000570  0.000003   0.000571   
    0.230500   -0.000894  0.000006   0.000897   
    0.255000   -0.001351  0.000013   0.001356   
    0.279500   -0.001960  0.000021   0.001968   
    0.304000   -0.002760  0.000033   0.002774   
    0.328500   -0.003791  0.000049   0.003811   
    0.353000   -0.005094  0.000068   0.005121   
    0.377500   -0.006707  0.000089   0.006742   
    0.402000   -0.008686  0.000112   0.008731   
    0.426500   -0.011069  0.000131   0.011122   
    0.451000   -0.013920  0.000139   0.013976   
    0.475500   -0.017304  0.000124   0.017354   
    0.500000   -0.021232  0.000067   0.021259  
    '''.split(),dtype=float).reshape(-1,4)

    assert match_values(Rs, biases_ref[:,0], tol=1e-5)
    assert match_values(biases_x, biases_ref[:,1], tol=1e-5)
    assert match_values(biases_y, biases_ref[:,2], tol=1e-5)
    assert match_values(biases_tot, biases_ref[:,3], tol=1e-5)

    # same as above, but from initialization
    tls4 = TargetLineSearch(
        structure = s,
        hessian = h,
        d = 0,
        R = 0.5,
        M = 7,
        bias_x0 = 0.0,
        bias_y0 = -0.5,
        bias_mix = 0.4,
        fit_kind = 'pf4',
        target_grid = grid,
        target_values = values,
        interpolate_kind = 'cubic',
    )
    biases_x, biases_y, biases_tot = tls4.compute_bias_of(Rs, verbose = False)
    assert match_values(Rs, biases_ref[:,0], tol=1e-5)
    assert match_values(biases_x, biases_ref[:,1], tol=1e-5)
    assert match_values(biases_y, biases_ref[:,2], tol=1e-5)
    assert match_values(biases_tot, biases_ref[:,3], tol=1e-5)

    # TODO: unit test maximize_sigma function manually
    # tls_test = TargetLinesearch(structure = s, hessian = h, d = 0)

    # test generation of W-sigma data: the error surface will be automatically extended, making the test a bit slower to run than most
    tls4.generate_W_sigma_data(
        sigma_max = 0.005,
        W_num = 5,
        sigma_num = 5,
        Gs = Gs_N200_M7)
    tls4.insert_sigma_data(0.0045)
    tls4.insert_W_data(0.05)
    W_ref = array('''
0.         0.03344238 0.05       0.06688476 0.10032714 0.13376953
0.         0.03344238 0.05       0.06688476 0.10032714 0.13376953
0.         0.03344238 0.05       0.06688476 0.10032714 0.13376953
0.         0.03344238 0.05       0.06688476 0.10032714 0.13376953
0.         0.03344238 0.05       0.06688476 0.10032714 0.13376953
0.         0.03344238 0.05       0.06688476 0.10032714 0.13376953
    '''.split(), dtype = float)
    S_ref = array('''
0.      0.      0.      0.      0.      0.     
0.00125 0.00125 0.00125 0.00125 0.00125 0.00125
0.0025  0.0025  0.0025  0.0025  0.0025  0.0025 
0.00375 0.00375 0.00375 0.00375 0.00375 0.00375
0.0045  0.0045  0.0045  0.0045  0.0045  0.0045 
0.005   0.005   0.005   0.005   0.005   0.005  
    '''.split(), dtype = float)
    E_ref = array('''
4.52112460e-06 1.25095641e-03 2.83723170e-03 5.15373644e-03  1.18319679e-02 2.12587458e-02
1.41120592e-02 5.42297785e-03 6.21844719e-03 8.08023571e-03  1.41361340e-02 2.30947758e-02
1.39965235e-02 9.46617238e-03 9.55472899e-03 1.09463151e-02  1.63852020e-02 2.48927799e-02
1.39985773e-02 1.32813681e-02 1.27452719e-02 1.37012997e-02  1.85817665e-02 2.66730377e-02
1.39992678e-02 1.56528175e-02 1.45600956e-02 1.53117739e-02  1.98755300e-02 2.77348900e-02
1.39514271e-02 1.72255235e-02 1.57640068e-02 1.63683030e-02  2.07282364e-02 2.84389483e-02
    '''.split(), dtype = float)
    assert match_values(tls4.W_mat, W_ref, tol = 1e-5)
    assert match_values(tls4.S_mat, S_ref, tol = 1e-5)
    assert match_values(tls4.E_mat, E_ref, tol = 1e-5)

    x1, y1 = tls4.maximize_sigma(epsilon= 0.01, verbose = False)
    x2, y2 = tls4.maximize_sigma(epsilon= 0.02, verbose = False)
    x3, y3 = tls4.maximize_sigma(epsilon= 0.03, verbose = False)
    x4, y4 = tls4.maximize_sigma(epsilon= 0.04, verbose = False)
    assert match_values([x1, y1], (0.05, 0.0025))
    assert match_values([x2, y2], (0.04172119081049441, 0.00625))
    assert match_values([x3, y3], (0.04172119081049441, 0.01))
    assert match_values([x4, y4], (0.11704833567346087, 0.015))
    assert not tls4.optimized
    tls4.optimize(epsilon= 0.05, verbose = False)
    x5, y5, eps5 = tls4.W_opt, tls4.sigma_opt, tls4.epsilon
    assert tls4.optimized
    assert match_values([x5, y5, eps5], (0.1337695264839553, 0.02, 0.05))
#end def
add_unit_test(test_targetlinesearch_class)
