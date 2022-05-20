#!/usr/bin/env python

from numpy import array, exp, nan, isnan, random, polyval, linspace
from pytest import raises

from testing import match_values, add_unit_test
from surrogate_classes import StructuralParameter
from surrogate_classes import ParameterStructureBase, ParameterStructure, ParameterHessian
from surrogate_classes import AbstractLineSearch, LineSearch
from surrogate_classes import AbstractTargetLineSearch, TargetLineSearch, TargetParallelLineSearch
from surrogate_classes import ParallelLineSearch, LineSearchIteration
from classes.assets import pos_H2O, elem_H2O, forward_H2O, backward_H2O, hessian_H2O, pes_H2O, hessian_real_H2O, get_structure_H2O, get_hessian_H2O
from classes.assets import pos_H2, elem_H2, forward_H2, backward_H2, hessian_H2, get_structure_H2, get_hessian_H2, get_surrogate_H2O
from classes.assets import params_GeSe, forward_GeSe, backward_GeSe, hessian_GeSe, elem_GeSe
from classes.assets import morse, Gs_N200_M7

# Test StructuralParameter class
def test_structuralparameter_class():
    # TODO: add features, add meaningful tests
    # test defaults
    p = StructuralParameter(0.74)
    assert p.value == 0.74
    assert p.kind == 'bond'
    assert p.label == 'r'
    assert p.unit == 'A'
    return True
#end def
add_unit_test(test_structuralparameter_class)


# Test ParameterStructureBase class
def test_parameterstructurebase_class():
    # test H2 (open; 1 parameter)
    s = ParameterStructureBase()

    # test inconsistent pos vector
    with raises(AssertionError):
        s.set_position([0.0, 0.0])
    #end with
    # test premature backward mapping
    with raises(AssertionError):
        s.backward()
    #end with
    assert not s.check_consistency()  # cannot be consistent without mapping functions

    # test backward mapping
    s.set_params([1.4])
    s.set_backward(backward_H2)  # pos should now be computed automatically
    assert match_values(s.pos, pos_H2, tol = 1e-5)

    assert not s.check_consistency()  # still not consistent, without forward mapping
    # test premature forward mapping
    with raises(AssertionError):
        s.forward()
    #end with

    s.set_position([0.0, 0.0, 0.0, 0.0, 0.0, 1.6])  # set another pos
    s.set_forward(forward_H2)  # then set forward mapping
    assert match_values(s.params, 1.6, tol = 1e-5)  # params computed automatically
    assert not s.check_consistency()  # pos is not translated in this order
    s.set_position([0.0, 0.0, 0.0, 0.0, 0.0, 1.6])  # this time translate position by default
    assert s.check_consistency()  # finally consistent
    assert s.check_consistency(params = [1.3])  # also consistent at another point
    assert s.check_consistency(pos = pos_H2)  # also consistent at another point
    assert s.check_consistency(pos = pos_H2 * 0.5, params = [0.7])  # consistent set of arguments
    assert not s.check_consistency(pos = pos_H2 * 0.5, params = [1.4])  # inconsistent set of arguments

    # test H2O (open; 2 parameters)
    s = ParameterStructureBase(pos = pos_H2O, forward = forward_H2O, elem = elem_H2O)
    params_ref = [0.95789707432, 104.119930724]
    assert match_values(s.params, params_ref, tol = 1e-5)

    # add backward mapping
    s.set_backward(backward_H2O)
    pos_ref = [[ 0.,          0.,          0.     ],
               [ 0.,          0.75545,     0.58895],
               [ 0.,         -0.75545,     0.58895]]
    assert match_values(s.pos, pos_ref, tol = 1e-5)
    assert s.check_consistency()

    # test another set of parameters
    s.backward([1.0, 120.0])
    pos2_ref = [[ 0.,          0.,          0. ],
                [ 0.,          0.8660254,   0.5],
                [ 0.,         -0.8660254,   0.5]]
    assert match_values(s.params, [1.0, 120.0], tol = 1e-5)
    assert match_values(s.pos, pos2_ref, tol = 1e-5)

    jac_ref = array('''
    0.          0.        
    0.          0.        
    0.          0.        
    0.          0.        
    0.8660254   0.00436329
    0.5        -0.00755752
    0.          0.        
   -0.8660254  -0.00436329
    0.5        -0.00755752
    '''.split(), dtype = float).reshape(-1, 2)
    assert match_values(jac_ref, s.jacobian())

    # test complete positional init of a periodic system (GeSe)
    s = ParameterStructureBase(
        forward_GeSe,  # forward
        backward_GeSe,  # backward
        None,  # pos
        None,  # axes
        elem_GeSe,  # elem
        params_GeSe,
        None,  # params_err
        True,  # periodic
        -10.0,  # value
        0.1,  # error
        'GeSe test',  # label
        'crystal',  # unit
        3,  # dim
    )
    pos_orig = s.pos
    s.shift_params([0.1,0.1,0.0,-0.1,0.05])
    params_ref = [4.360000, 4.050000, 0.414000, 0.456000, 0.610000 ]
    pos_ref = array('''
    0.414000 0.250000 0.456000
    0.914000 0.750000 0.544000
    0.500000 0.250000 0.390000
    0.000000 0.750000 0.610000
    '''.split(), dtype = float)
    axes_ref = array('''
    4.360000 0.000000 0.000000
    0.000000 4.050000 0.000000
    0.000000 0.000000 20.000000
    '''.split(), dtype = float)
    assert match_values(s.params, params_ref)
    assert match_values(s.pos, pos_ref)
    assert match_values(s.axes, axes_ref)
    dpos = pos_orig.flatten() - pos_ref
    s.shift_pos(dpos)
    params_ref2 = [4.360000, 4.050000, 0.414000, 0.556000, 0.560000]
    assert match_values(s.params, params_ref2)
    assert match_values(s.pos, pos_orig)
#end def
add_unit_test(test_parameterstructurebase_class)


# Test ParameterStructure class
def test_parameterstructure_class():
    # test empty
    s = ParameterStructure()
    if s.kind=='nexus':  # special tests for nexus Structure
        pass  # TODO
    #end if

    # test copy
    s1 = ParameterStructure(pos = pos_H2O, forward = forward_H2O, label = '1', elem = elem_H2O)
    s2 = s1.copy(params = s1.params * 1.5, label = '2')
    assert not match_values(s1.params, s2.params, expect_false = True)
    assert s1.label == '1'
    assert s2.label == '2'
    assert s1.forward_func == s2.forward_func
#end def
add_unit_test(test_parameterstructure_class)


def test_parameterhessian_class():
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


def test_abstractlinesearch_class():
    ls = AbstractLineSearch()  # test generation
    with raises(AssertionError):
        ls.get_x0()
    #end with
    with raises(AssertionError):
        ls.get_y0()
    #end with
    with raises(AssertionError):
        ls.search()
    #end with

    ls = AbstractLineSearch(
        grid = [0, 1, 2, 3, 4],
        values = [ 2, 1, 0, 1, 2],        
        fit_kind = 'pf3')
    x0 = ls.get_x0()
    y0 = ls.get_y0()
    x0_ref = [2.0000000000, 0.0]
    y0_ref = [0.3428571428, 0.0]
    fit_ref = [ 0.0,  4.28571429e-01, -1.71428571e+00,  2.05714286e+00]
    assert match_values(x0, x0_ref)
    assert match_values(y0, y0_ref)
    assert match_values(ls.fit, fit_ref)

    # test setting wrong number of values
    with raises(AssertionError):
        ls.set_values(ls.values[:-2])
    #end with

    # test _search method
    x2, y2, pf2 = ls._search(2 * ls.grid, 2 * ls.values, None)
    assert match_values(x2, 2 * x0_ref[0])
    assert match_values(y2, 2 * y0_ref[0])
    x3, y3, pf3 = ls._search(ls.grid, ls.values, fit_kind = 'pf2')
    assert match_values(x3, [2.0])
    assert match_values(y3, [0.34285714285])
    assert match_values(pf3, [ 0.42857143, -1.71428571,  2.05714286])
    # TODO: more tests
#end def
add_unit_test(test_abstractlinesearch_class)


def test_abstracttargetlinesearch_class():

    # generate reference potential
    p = [1.0, 0.5, 0.5, 0.0]
    grid = linspace(-0.5, 0.5, 101)
    values = morse(p, grid + 1.0)

    ls = AbstractTargetLineSearch(
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
add_unit_test(test_abstracttargetlinesearch_class)


# test LineSearch
def test_linesearch_class():
    results = []
    s = get_structure_H2O()
    h = get_hessian_H2O()

    with raises(TypeError):
        ls_d0 = LineSearch()
    #end with
    with raises(TypeError):
        ls_d0 = LineSearch(s)
    #end with
    with raises(TypeError):
        ls_d0 = LineSearch(s, h)
    #end with
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
    assert match_values(gridR_d0, gridR_d0_ref)
    assert match_values(gridR_d1, gridR_d1_ref)
    assert match_values(gridW_d0, gridW_d0_ref)
    assert match_values(gridW_d1, gridW_d1_ref)

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
add_unit_test(test_linesearch_class)


# test TargetLineSearch class
def test_targetlinesearch_class():

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
4.52112460e-06 1.25095641e-03 2.83723170e-03 5.15373644e-03 1.18319679e-02 2.12587458e-02
9.03844597e-03 5.42297785e-03 6.21844719e-03 8.08023571e-03 1.41361340e-02 2.30947758e-02
9.04268709e-03 9.46617238e-03 9.55472899e-03 1.09463151e-02 1.63852020e-02 2.48927799e-02
1.05408674e-02 1.32813681e-02 1.27452719e-02 1.37012997e-02 1.85817665e-02 2.66730377e-02
1.04723739e-02 1.56528175e-02 1.45600956e-02 1.53117739e-02 1.98755300e-02 2.77348900e-02
1.04180628e-02 1.72255235e-02 1.57640068e-02 1.63683030e-02 2.07282364e-02 2.84389483e-02
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


def test_parallellinesearch_class():
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


# test TargetParallelLineSearch class
def test_targetparallellinesearch_class():
    s = get_structure_H2O()
    h = get_hessian_H2O()
    srg = TargetParallelLineSearch(
        structure = s,
        hessian = h,
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
    assert match_values(srg.ls(0).grid, grid0_ref)
    assert match_values(srg.ls(1).grid, grid1_ref)
    assert match_values(params0, params0_ref)
    assert match_values(params1, params1_ref)

    values0 = [pes_H2O(p) for p in params0]
    values1 = [pes_H2O(p) for p in params1]
    values0_ref = [0.34626073187519557, -0.3652007349305562, -0.49999956687591435, -0.4396197672411492, -0.33029670717647247]
    values1_ref = [-0.31777610098060916, -0.4523302132096337, -0.49999956687591435, -0.4456834605043901, -0.2662664861469014]
    assert match_values(values0, values0_ref)
    assert match_values(values1, values1_ref)

    srg.load_results(values = [values0, values1], set_target = True)

    bias_d, bias_p = srg.compute_bias(windows = [0.1, 0.05])
    bias_d_ref = [-0.0056427, -0.0003693]
    bias_p_ref = [-0.00520237, -0.00221626]
    assert match_values(bias_d, bias_d_ref, tol = 1e-5)
    assert match_values(bias_p, bias_p_ref, tol = 1e-5)

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
    srg.optimize(windows = [0.1,0.05], noises = [0.02, 0.02], Gs = Gs_N200_M7.reshape(2, -1, 5), W_num = 5, sigma_num = 5)

    assert match_values(srg.windows, [0.1,  0.05])
    assert match_values(srg.noises,  [0.02, 0.02])
    assert match_values(srg.error_d, [0.03312047, 0.1099758])
    assert match_values(srg.error_p, [0.05657029, 0.10617361])

    #2: thermal
    srg.optimize(temperature = 0.0001, Gs = Gs_N200_M7.reshape(2, -1, 5), W_num = 5, sigma_num = 5, verbose = False, fix_res = False)
    assert match_values(srg.windows,   [0.1034483548381337, 0.06556247311750507])
    assert match_values(srg.noises,    [0.0025862088709533, 0.00245859274190644])
    assert match_values(srg.error_d,   [0.00922068, 0.01366444])
    assert match_values(srg.error_p,   [0.01136238, 0.01459201])
    assert match_values(srg.epsilon_d, [0.009666659286797805, 0.015252627798340353])
    assert srg.epsilon_p is None
    assert srg.ls(0).E_mat.shape == (5, 5)
    assert srg.ls(1).E_mat.shape == (5, 5)

    srg.optimize(temperature = 0.0002, verbose = False, fix_res = True)
    assert match_values(srg.windows,   [0.1034483548381337,   0.0491718548381288])
    assert match_values(srg.noises,    [0.005172417741906685, 0.0032781236558752534])
    assert match_values(srg.error_d,   [0.0127679 , 0.02093523])
    assert match_values(srg.error_p,   [0.01622812, 0.02110696])
    assert match_values(srg.epsilon_d, [0.013670720666229288, 0.02157047309424181])
    assert srg.epsilon_p is None
    assert srg.ls(0).E_mat.shape == (6, 5)
    assert srg.ls(1).E_mat.shape == (6, 5)

    #3: epsilon_d
    with raises(AssertionError):  # too low tolerances
        srg.optimize(epsilon_d = [0.001, 0.001], verbose = False)
    #end with
    srg.optimize(epsilon_d = [0.008, 0.007], verbose = False)
    assert match_values(srg.windows,   [0.1034483548381337, 0.032781236558752536])
    assert match_values(srg.noises,    [0.0012931044354766712, 0.0008195309139688133])
    assert match_values(srg.error_d,   [0.00741869, 0.00670662])
    assert match_values(srg.error_p,   [0.00821476, 0.00810403])
    assert match_values(srg.epsilon_d, [0.008, 0.007])

    #4: epsilon_p with thermal
    srg.optimize(epsilon_p = [0.01, 0.015], kind='thermal', T0 = 0.00001, dT = 0.000005, verbose = False)
    assert match_values(srg.windows,   [0.1034483548381337, 0.04097654569844067])
    assert match_values(srg.noises,    [0.0012931044354766712, 0.0016390618279376267])
    assert match_values(srg.error_d,   [0.00741869, 0.01167165])
    assert match_values(srg.error_p,   [0.00944918, 0.01285586])
    assert match_values(srg.epsilon_d, [0.008087707415388121, 0.01276126397847382])
    assert match_values(srg.epsilon_p, [0.01, 0.015])

    #5 epsilon_p with ls
    srg.optimize(epsilon_p = [0.02, 0.05], kind='ls', verbose = False)

    assert match_values(srg.windows,   [0.1034483548381337, 0.04097654569844067])
    assert match_values(srg.noises,    [8.081902721729195e-05, 0.006556247311750507])
    assert match_values(srg.error_d,   [0.0057126 , 0.04470828])
    assert match_values(srg.error_p,   [0.01941638, 0.04404034])
    assert match_values(srg.epsilon_d, [0.005723591908711029, 0.04878897098061215])
    assert match_values(srg.epsilon_p, [0.02, 0.05])

#end def
add_unit_test(test_targetparallellinesearch_class)


# test LineSearchIteration class
def test_linesearchiteration_class():
    from shutil import rmtree
    s = get_structure_H2O()
    s.shift_params([0.2, -0.2])
    params_ref = s._forward(pos_H2O)
    h = get_hessian_H2O()
    # must make global for pickling
    from classes.assets import job_H2O_pes, analyze_H2O_pes

    # test deterministic line-search iteration
    test_dir = 'tmp/test_pls_h2O/'
    rmtree(test_dir, ignore_errors = True)
    lsi = LineSearchIteration(
        path = test_dir,
        hessian = h,
        structure = s,
        job_func = job_H2O_pes,
        analyze_func = analyze_H2O_pes,
        windows = [0.05, 1.0],
        load = False)
    job_data = lsi.generate_jobs()
    lsi.load_results(job_data = job_data)
    lsi.propagate(write = True)
    assert match_values(lsi.pls_list[-1].structure.params, [  0.89725537, 104.12804938])
    # second iteration
    job_data = lsi.generate_jobs()
    lsi.load_results(job_data = job_data)
    lsi.propagate(write = True)
    assert match_values(lsi.pls_list[-1].structure.params, [  0.93244294, 104.1720672 ])
    # third iteration
    job_data = lsi.generate_jobs()
    lsi.load_results(job_data = job_data)
    lsi.propagate(write = False)
    assert match_values(lsi.pls_list[-1].structure.params, [  0.93703957, 104.20617541])
    # start over and load until second iteration
    lsi = LineSearchIteration(path = test_dir, load = True)
    assert len(lsi.pls_list) == 2
    lsi.propagate(write = False)
    lsi.generate_jobs()
    lsi.load_results(job_data = job_data)
    lsi.propagate(write = False)
    assert match_values(lsi.pls_list[-1].structure.params, [  0.93703957, 104.20617541])
    rmtree(test_dir)

    # test starting from surrogate
    test_dir = 'tmp/test_pls_srg_H2O/'
    rmtree(test_dir, ignore_errors = True)
    srg = get_surrogate_H2O()
    srg.optimize(windows = [0.1, 0.05], noises = [0.005, 0.005], M = 5, Gs = Gs_N200_M7.reshape(2, -1, 5))
    lsi = LineSearchIteration(
        path = test_dir,
        surrogate = srg,
        job_func = job_H2O_pes,
        analyze_func = analyze_H2O_pes)
    job_data = lsi.generate_jobs()
    lsi.load_results(job_data = job_data)
    lsi.propagate(write = True)
    grid0_ref = [-0.432306, -0.216153, 0., 0.216153, 0.432306]
    grid1_ref = [-0.482330, -0.241165, 0., 0.241165, 0.482330]
    assert match_values(lsi.pls(0).ls(0).grid, grid0_ref, tol = 1e-5)
    assert match_values(lsi.pls(0).ls(1).grid, grid1_ref, tol = 1e-5)
    assert match_values(lsi.pls().ls(0).grid, grid0_ref, tol = 1e-5)
    assert match_values(lsi.pls().ls(1).grid, grid1_ref, tol = 1e-5)
    rmtree(test_dir)
#end def
add_unit_test(test_linesearchiteration_class)
