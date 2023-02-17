#!/usr/bin/env python

from numpy import array, exp, nan, isnan, random, polyval, linspace
from pytest import raises
from surrogate_classes import match_to_tol

from assets import pos_H2O, elem_H2O, forward_H2O, backward_H2O, hessian_H2O, pes_H2O, hessian_real_H2O, get_structure_H2O, get_hessian_H2O
from assets import pos_H2, elem_H2, forward_H2, backward_H2, hessian_H2, get_structure_H2, get_hessian_H2, get_surrogate_H2O
from assets import params_GeSe, forward_GeSe, backward_GeSe, hessian_GeSe, elem_GeSe
from assets import morse, Gs_N200_M7


# test TargetParallelLineSearch class
def test_targetparallellinesearch_class():
    from surrogate_classes import TargetParallelLineSearch
    from surrogate_classes import ParameterSet
    s = get_structure_H2O()
    h = get_hessian_H2O()
    srg = TargetParallelLineSearch(
        structure = s,
        hessian = h,
        mode = 'pes',
        targets = [0.01, -0.01], # NOTE!
        M = 5,
        window_frac = 0.1)

    params0 = srg.get_shifted_params(0)
    params1 = srg.get_shifted_params(1)

    grid0_ref = [-0.4396967 , -0.21984835,  0.,          0.21984835,  0.4396967 ]
    grid1_ref = [-0.55231563, -0.27615782,  0.,          0.27615782,  0.55231563]
    params0_ref = array('''
    0.54298682 103.97438807
    0.75044195 104.0471594 
    0.95789707 104.11993072
    1.1653522  104.19270205
    1.37280733 104.26547338
    '''.split(), dtype = float)
    params1_ref = array('''
    1.14071738 103.59875005
    1.04930723 103.85934039
    0.95789707 104.11993072
    0.86648692 104.38052106
    0.77507677 104.64111139
    '''.split(), dtype = float)
    assert match_to_tol(srg.ls(0).grid, grid0_ref)
    assert match_to_tol(srg.ls(1).grid, grid1_ref)
    assert match_to_tol(params0, params0_ref)
    assert match_to_tol(params1, params1_ref)

    values0 = [pes_H2O(ParameterSet(p))[0] for p in params0]
    values1 = [pes_H2O(ParameterSet(p))[0] for p in params1]
    values0_ref = [0.34626073187519557, -0.3652007349305562, -0.49999956687591435, -0.4396197672411492, -0.33029670717647247]
    values1_ref = [-0.31777610098060916, -0.4523302132096337, -0.49999956687591435, -0.4456834605043901, -0.2662664861469014]
    assert match_to_tol(values0, values0_ref)
    assert match_to_tol(values1, values1_ref)

    srg.load_results(values = [values0, values1], set_target = True)

    bias_d, bias_p = srg.compute_bias(windows = [0.1, 0.05])
    bias_d_ref = [-0.0056427, -0.0003693]
    bias_p_ref = [-0.00520237, -0.00221626]
    assert match_to_tol(bias_d, bias_d_ref, tol = 1e-5)
    assert match_to_tol(bias_p, bias_p_ref, tol = 1e-5)

    # test optimization
    #1: windows, noises
    with raises(AssertionError):
        srg.optimize(windows = [0.1,0.05], Gs = Gs_N200_M7.reshape(2, -1, 5), W_num = 5, sigma_num = 5, verbose = False)
    #end with
    with raises(AssertionError):
        srg.optimize(noises = [0.02,0.02], Gs = Gs_N200_M7.reshape(2, -1, 5), W_num = 5, sigma_num = 5, verbose = False)
    #end with
    with raises(AssertionError): # too large W
        srg.optimize(windows = [0.2,0.4], noises = [0.02, 0.02], Gs = Gs_N200_M7.reshape(2, -1, 5), W_num = 5, sigma_num = 5)
    #end with
    srg.optimize(windows = [0.1,0.05], noises = [0.02, 0.02], Gs = Gs_N200_M7.reshape(2, -1, 5), W_num = 5, sigma_num = 5, verbose = False)

    assert match_to_tol(srg.windows, [0.1,  0.05])
    assert match_to_tol(srg.noises,  [0.02, 0.02])
    assert match_to_tol(srg.error_d, [0.03312047, 0.1099758])
    assert match_to_tol(srg.error_p, [0.05657029, 0.10617361])

    #2: thermal
    srg.reoptimize(temperature = 0.0001, Gs = Gs_N200_M7.reshape(2, -1, 5), W_num = 5, sigma_num = 5, verbose = False, fix_res = False)
    assert match_to_tol(srg.windows,   [0.1034483548381337, 0.06556247311750507])
    assert match_to_tol(srg.noises,    [0.0025862088709533, 0.00245859274190644])
    assert match_to_tol(srg.error_d,   [0.00922068, 0.01366444])
    assert match_to_tol(srg.error_p,   [0.01136238, 0.01459201])
    assert match_to_tol(srg.epsilon_d, [0.009666659286797805, 0.015252627798340353])
    assert srg.epsilon_p is None
    assert srg.ls(0).E_mat.shape == (5, 5)
    assert srg.ls(1).E_mat.shape == (5, 5)

    srg.reoptimize(temperature = 0.0002, verbose = False, fix_res = True)
    assert match_to_tol(srg.windows,   [0.1034483548381337,   0.0491718548381288])
    assert match_to_tol(srg.noises,    [0.005172417741906685, 0.0032781236558752534])
    assert match_to_tol(srg.error_d,   [0.0127679 , 0.02093523])
    assert match_to_tol(srg.error_p,   [0.01622812, 0.02110696])
    assert match_to_tol(srg.epsilon_d, [0.013670720666229288, 0.02157047309424181])
    assert srg.epsilon_p is None
    assert srg.ls(0).E_mat.shape == (6, 5)
    assert srg.ls(1).E_mat.shape == (6, 5)

    #3: epsilon_d
    with raises(AssertionError):  # too low tolerances
        srg.reoptimize(epsilon_d = [0.001, 0.001], verbose = False)
    #end with
    srg.reoptimize(epsilon_d = [0.008, 0.007], verbose = False)

    assert match_to_tol(srg.windows,   [0.1034483548381337, 0.061464818547661004])
    assert match_to_tol(srg.noises,    [0.0012931044354766712, 0.0011268550067071183])
    assert match_to_tol(srg.error_d,   [0.00741869, 0.00666114])
    assert match_to_tol(srg.error_p,   [0.0082076, 0.0080603])
    assert match_to_tol(srg.epsilon_d, [0.008, 0.007])

    #4: epsilon_p with thermal
    srg.reoptimize(epsilon_p = [0.01, 0.015], kind='thermal', T0 = 0.00001, dT = 0.000005, verbose = False)
    assert match_to_tol(srg.windows,   [0.1034483548381337, 0.05736716397781694])
    assert match_to_tol(srg.noises,    [0.0012931044354766712, 0.0020488272849220335])
    assert match_to_tol(srg.error_d,   [0.00741869, 0.01224184])
    assert match_to_tol(srg.error_p,   [0.00963971, 0.0133939])
    assert match_to_tol(srg.epsilon_d, [0.008190424524106486, 0.012923337118875344])
    assert match_to_tol(srg.epsilon_p, [0.01, 0.015])

    #5 epsilon_p with ls
    srg.reoptimize(epsilon_p = [0.02, 0.05], kind='ls', verbose = False)
    assert match_to_tol(srg.windows,   [0.09051731048336698, 0.061464818547661004])
    assert match_to_tol(srg.noises,    [0.0006465522177383356, 0.007375778225719321])
    assert match_to_tol(srg.error_d,   [0.00677768, 0.04134956])
    assert match_to_tol(srg.error_p,   [0.01943834, 0.04092298])
    assert match_to_tol(srg.epsilon_d, [0.007278290487659847, 0.0440791914501621])
    assert match_to_tol(srg.epsilon_p, [0.02, 0.05])

#end def

