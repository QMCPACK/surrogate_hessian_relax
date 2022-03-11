#!/usr/bin/env python

from numpy import array,exp,nan,isnan,random, polyval, linspace

from testing import add_unit_test, match_values, resolve

from surrogate_classes import ParameterStructureBase, ParameterStructure, ParameterHessian
from surrogate_classes import AbstractLineSearch, LineSearch
from surrogate_classes import AbstractTargetLineSearch, TargetLineSearch, TargetParallelLineSearch
from surrogate_classes import ParallelLineSearch, LineSearchIteration
from classes.assets import pos_H2O, elem_H2O, forward_H2O, backward_H2O, hessian_H2O, pes_H2O
from classes.assets import pos_H2, elem_H2, forward_H2, backward_H2, hessian_H2
from classes.assets import params_GeSe, forward_GeSe, backward_GeSe, hessian_GeSe
from classes.assets import morse

# Test ParameterStructureBase class
def test_parameterstructurebase_class():
    results = []

    # test H2 (open; 1 parameter)
    pm = ParameterStructureBase()

    # test inconsistent pos vector
    try:
        pm.set_pos([0.0, 0.0])
        results.append(False)
    except AssertionError:
        results.append(True)
    #end try

    # test premature backward mapping
    try:
        pm.backward()
        results.append(False)
    except AssertionError:
        results.append(True)
    #end try

    # test backward mapping
    pm.set_params([1.4])
    pm.set_backward(backward_H2)  # pos should be computed automatically
    results.append(match_values(pm.pos, pos_H2, tol=1e-5))

    results.append(not pm.check_consistency())  # not yet consistent
    # test premature forward mapping
    try:
        pm.forward()
        results.append(False)
    except AssertionError:
        results.append(True)
    #end try

    pm.set_forward(forward_H2)
    results.append(match_values(pm.params, 1.4, tol=1e-5))  # params computed automatically
    results.append(pm.check_consistency())  # finally consistent

    # test H2O (open; 2 parameters)
    pm = ParameterStructureBase(pos = pos_H2O, forward = forward_H2O)
    params_ref = [0.95789707432, 104.119930724]
    results.append(match_values(pm.params, params_ref, tol=1e-5))

    # add backward mapping
    pm.set_backward(backward_H2O)
    pos_ref = [[ 0.,          0.,          0.     ],
               [ 0.,          0.75545,     0.58895],
               [ 0.,         -0.75545,     0.58895]]
    results.append(match_values(pm.pos, pos_ref, tol=1e-5))
    results.append(pm.check_consistency())

    # test another set of parameters
    pos2 = pm.backward([1.0, 120.0])
    pos2_ref = [[ 0.,          0.,          0. ],
                [ 0.,          0.8660254,   0.5],
                [ 0.,         -0.8660254,   0.5]]
    results.append(match_values(pos2, pos2_ref, tol = 1e-5))

    # test GeSe (periodic; 5 parameters)

    # test when only mappings are provided

    return resolve(results)
#end def
add_unit_test(test_parameterstructurebase_class)

# Test ParameterStructure class
def test_parameterstructure_class():
    results = []

    # test empty
    s = ParameterStructure()
    if s.kind=='nexus':  # special tests for nexus Structure
        pass  # TODO
    #end if

    # test copy
    s1 = ParameterStructure(pos = pos_H2O, forward = forward_H2O, label = '1')
    s2 = s1.copy(params = s1.params * 1.5, label = '2')
    results.append(not match_values(s1.params, s2.params, expect_false = True))
    results.append(s1.label == '1')
    results.append(s2.label == '2')
    results.append(s1.forward_func == s2.forward_func)

    return resolve(results)
#end def
add_unit_test(test_parameterstructure_class)


def test_parameterhessian_class():
    results = []

    structure = ParameterStructure(
        forward = forward_H2O,
        backward = backward_H2O,
        pos = pos_H2O)
    hessian = ParameterHessian(hessian_H2O)
    Lambda = hessian.Lambda
    dire = hessian.get_directions()
    dir0 = hessian.get_directions(0)
    dir1 = hessian.get_directions(1)
    Lambda_ref = array([1.07015621, 0.42984379])
    dire_ref = array([[ 0.94362832,  0.33100694],
                      [-0.33100694,  0.94362832]])
    results.append(match_values(dire, dire_ref))
    results.append(match_values(dir0, dire_ref[0]))
    results.append(match_values(dir1, dire_ref[1]))
    results.append(match_values(Lambda, Lambda_ref))
    # TODO: test conversions
    # TODO: test dummy init
    try:
        dir2 = hessian.get_directions(2)
        results.append(False)
    except IndexError:
        results.append(True)
    #end try

    return resolve(results)
#end def
add_unit_test(test_parameterhessian_class)


def test_abstractlinesearch_class():
    results = []
    ls = AbstractLineSearch()  # test generation
    try:
        ls.get_x0()
    except AssertionError:
        results.append(True)
    #end try
    try:
        ls.get_y0()
    except AssertionError:
        results.append(True)
    #end try
    try:
        ls.search()
    except AssertionError:
        results.append(True)
    #end try

    ls = AbstractLineSearch(
        grid = [0, 1, 2, 3, 4],
        values = [ 2, 1, 0, 1, 2],        
        fit_kind = 'pf3')
    x0 = ls.get_x0()
    y0 = ls.get_y0()
    x0_ref = [2.0000000000, 0.0]
    y0_ref = [0.3428571428, 0.0]
    fit_ref = [ 0.0,  4.28571429e-01, -1.71428571e+00,  2.05714286e+00]
    results.append(match_values(x0, x0_ref))
    results.append(match_values(y0, y0_ref))
    results.append(match_values(ls.fit, fit_ref))

    # TODO: more tests
    return resolve(results)
#end def
add_unit_test(test_abstractlinesearch_class)


def test_abstracttargetlinesearch_class():
    results = []

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

    return resolve(results)
#end def
add_unit_test(test_abstracttargetlinesearch_class)


# test LineSearch
def test_linesearch_class():
    results = []
    structure = ParameterStructure(
        forward = forward_H2O,
        backward = backward_H2O,
        pos = pos_H2O)
    hessian = ParameterHessian(hessian_H2O)

    try:
        ls_d0 = LineSearch()
        results.append(False)
    except TypeError:
        results.append(True)
    #end try
    try:
        ls_d0 = LineSearch(structure)
        results.append(False)
    except TypeError:
        results.append(True)
    #end try
    try:
        ls_d0 = LineSearch(structure, hessian)
        results.append(False)
    except TypeError:
        results.append(True)
    #end try
    try:
        ls_d0 = LineSearch(structure, hessian, d = 1)
        results.append(False)
    except AssertionError:
        results.append(True)
    #end try

    ls_d0 = LineSearch(structure, hessian, d = 0, R = 1.3)
    ls_d1 = LineSearch(structure, hessian, d = 1, W = 3.0)
    # test grid generation
    gridR_d0 = ls_d0._make_grid_R(1.3, M = 9)
    gridW_d0 = ls_d0._make_grid_W(3.0, M = 7)
    gridR_d1 = ls_d1._make_grid_R(1.3, M = 7)
    gridW_d1 = ls_d1._make_grid_W(3.0, M = 9)
    gridR_d0_ref = array('-1.3   -0.975 -0.65  -0.325  0.     0.325  0.65   0.975  1.3'.split(),dtype=float)
    gridW_d0_ref = array('-2.36783828 -1.57855885 -0.78927943  0.          0.78927943  1.57855885 2.36783828'.split(),dtype=float)
    gridR_d1_ref = array('-1.3        -0.86666667 -0.43333333  0.          0.43333333  0.86666667  1.3'.split(),dtype=float)
    gridW_d1_ref = array('-3.73611553 -2.80208665 -1.86805777 -0.93402888  0.          0.93402888  1.86805777  2.80208665  3.73611553'.split(),dtype=float)
    results.append(match_values(gridR_d0, gridR_d0_ref))
    results.append(match_values(gridR_d1, gridR_d1_ref))
    results.append(match_values(gridW_d0, gridW_d0_ref))
    results.append(match_values(gridW_d1, gridW_d1_ref))

    # try negative grid parameters
    try:
        grid = ls_d0._make_grid_R(-1.0e-2, 7)
        results.append(False)
    except AssertionError:
        results.append(True)
    #end try
    try:
        grid = ls_d0._make_grid_W(0.0, 7)
        results.append(False)
    except AssertionError:
        results.append(True)
    #end try

    return resolve(results)
#end def
add_unit_test(test_linesearch_class)


# test TargetLineSearch class
def test_targetlinesearch_class():

    results = []
    # generate reference potential
    p = [1.0, 0.5, 0.5, 0.0]
    grid = linspace(-0.51, 0.51, 21)
    values = morse(p, grid + 1.0)

    s = ParameterStructure(forward = forward_H2, backward = backward_H2, pos = pos_H2)
    h = ParameterHessian(hessian_H2)

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
    results.append(match_values(Rs, biases_ref[:,0], tol=1e-5))
    results.append(match_values(biases_x, biases_ref[:,1], tol=1e-5))
    results.append(match_values(biases_y, biases_ref[:,2], tol=1e-5))
    results.append(match_values(biases_tot, biases_ref[:,3], tol=1e-5))

    # bias_mix = 0.4, pf4, cubic
    tls0.set_target(grid, values, interpolate_kind = 'cubic')
    biases_x, biases_y, biases_tot = tls0.compute_bias_of(Rs, fit_kind = 'pf4', bias_mix = 0.4, M = 7, verbose = False)
    biases_ref = array('''
    0.010000   -0.000005  -0.000000  0.000005   
    0.034500   -0.000005  -0.000000  0.000005   
    0.059000   -0.000005  -0.000001  0.000005   
    0.083500   -0.000016  -0.000001  0.000016   
    0.108000   -0.000045  -0.000000  0.000045   
    0.132500   -0.000096  0.000000   0.000096   
    0.157000   -0.000190  0.000001   0.000190   
    0.181500   -0.000342  0.000002   0.000343   
    0.206000   -0.000570  0.000003   0.000571   
    0.230500   -0.000895  0.000006   0.000897   
    0.255000   -0.001351  0.000013   0.001356   
    0.279500   -0.001960  0.000021   0.001968   
    0.304000   -0.002761  0.000033   0.002774   
    0.328500   -0.003791  0.000049   0.003811   
    0.353000   -0.005094  0.000068   0.005121   
    0.377500   -0.006707  0.000089   0.006742   
    0.402000   -0.008687  0.000112   0.008732   
    0.426500   -0.011070  0.000131   0.011122   
    0.451000   -0.013921  0.000139   0.013976   
    0.475500   -0.017304  0.000124   0.017354   
    0.500000   -0.021233  0.000067   0.021259   
    '''.split(),dtype=float).reshape(-1,4)
    results.append(match_values(Rs, biases_ref[:,0], tol=1e-5))
    results.append(match_values(biases_x, biases_ref[:,1], tol=1e-5))
    results.append(match_values(biases_y, biases_ref[:,2], tol=1e-5))
    results.append(match_values(biases_tot, biases_ref[:,3], tol=1e-5))

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
    results.append(match_values(Rs, biases_ref[:,0], tol=1e-5))
    results.append(match_values(biases_x, biases_ref[:,1], tol=1e-5))
    results.append(match_values(biases_y, biases_ref[:,2], tol=1e-5))
    results.append(match_values(biases_tot, biases_ref[:,3], tol=1e-5))

    return resolve(results)
#end def
add_unit_test(test_targetlinesearch_class)


def test_parallellinesearch_class():
    results = []

    s = ParameterStructure(
        forward = forward_H2O,
        backward = backward_H2O,
        pos = pos_H2O)
    s.shift_params([0.2, 0.2])
    h = ParameterHessian(hessian_H2O)
    pls = ParallelLineSearch(
        hessian = h,
        structure = s,
        M = 9,
        window_frac = 0.1,
        noise_frac = 0.0)
    results.append(not pls.protected)
    results.append(not pls.generated)
    results.append(not pls.loaded)
    results.append(not pls.calculated)

    ls0 = pls.ls_list[0]
    ls1 = pls.ls_list[1]

    # test grid
    ls0_grid_ref = array('''-0.4396967  -0.32977252 -0.21984835 -0.10992417  0.          0.10992417
  0.21984835  0.32977252  0.4396967 '''.split(), dtype=float)
    ls1_grid_ref = array('''-0.55231563 -0.41423672 -0.27615782 -0.13807891  0.          0.13807891
  0.27615782  0.41423672  0.55231563'''.split(), dtype=float)
    results.append(match_values(ls0.grid, ls0_grid_ref))
    results.append(match_values(ls1.grid, ls1_grid_ref))
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
    results.append(match_values(params0, params0_ref))
    results.append(match_values(params1, params1_ref))
    # test PES
    values0 = [pes_H2O(params) for params in params0]
    values1 = [pes_H2O(params) for params in params1]
    values0_ref = array('''-0.35429145 -0.4647814  -0.49167476 -0.47112498 -0.42546898 -0.36820753
 -0.30724027 -0.24695829 -0.18959033'''.split(), dtype = float)
    values1_ref = array('''-0.3056267  -0.3616872  -0.40068042 -0.42214136 -0.42546898 -0.40989479
 -0.37444477 -0.31789329 -0.23870716'''.split(), dtype = float)
    results.append(match_values(values0, values0_ref))
    results.append(match_values(values1, values1_ref))

    # test loading without function
    pls.load_results()
    results.append(not pls.loaded)
    # manually enter values
    pls.load_results(values=[values0, values1])
    results.append(pls.loaded)

    ls0_x0_ref = -0.19600534, 0.0
    ls0_y0_ref = -0.48854587, 0.0
    ls1_x0_ref = -0.04318508, 0.0
    ls1_y0_ref = -0.42666697, 0.0
    results.append(match_values(ls0.get_x0(), ls0_x0_ref))
    results.append(match_values(ls0.get_y0(), ls0_y0_ref))
    results.append(match_values(ls1.get_x0(), ls1_x0_ref))
    results.append(match_values(ls1.get_y0(), ls1_y0_ref))
  
    next_params_ref = [0.98723545, 104.21430094]
    results.append(match_values(pls.get_next_params(), next_params_ref))

    return resolve(results)
#end def
add_unit_test(test_parallellinesearch_class)


# test TargetParallelLineSearch class
def test_targetparallellinesearch_class():
    results = []
    
    s = ParameterStructure(
        forward = forward_H2O,
        backward = backward_H2O,
        pos = pos_H2O)
    h = ParameterHessian(hessian_H2O)
    srg = TargetParallelLineSearch(
        structure = s,
        hessian = h,
        targets = [0.01,-0.01],
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
    results.append(match_values(srg.ls(0).grid, grid0_ref))
    results.append(match_values(srg.ls(1).grid, grid1_ref))
    results.append(match_values(params0, params0_ref))
    results.append(match_values(params1, params1_ref))

    values0 = [pes_H2O(p) for p in params0]
    values1 = [pes_H2O(p) for p in params1]
    values0_ref = [0.34626073187519557, -0.3652007349305562, -0.49999956687591435, -0.4396197672411492, -0.33029670717647247]
    values1_ref = [-0.31777610098060916, -0.4523302132096337, -0.49999956687591435, -0.4456834605043901, -0.2662664861469014]
    results.append(match_values(values0, values0_ref))
    results.append(match_values(values1, values1_ref))

    srg.load_results(values = [values0, values1], set_target = True)

    bias_d, bias_p = srg.compute_bias(windows = [0.1, 0.05])
    bias_d_ref = [-0.0157014,  0.0096304]
    bias_p_ref = [-0.01800402,  0.00389025]
    results.append(match_values(bias_d, bias_d_ref))
    results.append(match_values(bias_p, bias_p_ref))

    # TODO: resampling
    

    return resolve(results)
#end def
add_unit_test(test_targetparallellinesearch_class)


# test LineSearchIteration class

# defined here to make functions global for pickling
s = ParameterStructure(
    forward = forward_H2O,
    backward = backward_H2O,
    pos = pos_H2O,
    elem = ['O', 'H', 'H'])
s.shift_params([0.2, -0.2])
h = ParameterHessian(hessian_H2O)
params_ref = s.forward(pos_H2O)

def job_H2O_pes(structure, path, sigma, **kwargs):
    p = structure.params
    value = pes_H2O(p)
    return [(path, value, sigma)]
#end def
def analyze_H2O_pes(path, job_data = None, **kwargs):
    for row in job_data:
        if path == row[0]:
            return row[1], row[2]
        #end if
    #end for
    return None
#end def

def test_linesearchiteration_class():
    results = []

    # test deterministic line-search iteration
    test_dir = 'tmp/test_pls_h2O/'
    lsi = LineSearchIteration(
        path = test_dir,
        hessian = h,
        structure = s,
        job_func = job_H2O_pes,
        analyze_func = analyze_H2O_pes,
        windows = [0.05, 1.0],
        no_load = True,
    )
    job_data = lsi.generate_jobs()
    lsi.load_results(job_data = job_data)
    lsi.propagate(write = True)
    results.append(match_values(lsi.pls_list[-1].structure.params, [  0.89725537, 104.12804938]))

    # second iteration
    job_data = lsi.generate_jobs()
    lsi.load_results(job_data = job_data)
    lsi.propagate(write = True)
    results.append(match_values(lsi.pls_list[-1].structure.params, [  0.93244294, 104.1720672 ]))
    
    # third iteration
    job_data = lsi.generate_jobs()
    lsi.load_results(job_data = job_data)
    lsi.propagate(write = False)
    results.append(match_values(lsi.pls_list[-1].structure.params, [  0.93703957, 104.20617541]))

    # start over and load until second iteration
    lsi = LineSearchIteration(path = test_dir)
    results.append(len(lsi.pls_list) == 2)

    lsi.propagate(write = False)
    lsi.generate_jobs()
    lsi.load_results(job_data = job_data)
    lsi.propagate(write = False)
    results.append(match_values(lsi.pls_list[-1].structure.params, [  0.93703957, 104.20617541]))

    # TODO: test starting from surrogate
    # TODO: test stochastic line-search

    # remove test directory
    from shutil import rmtree
    rmtree(test_dir)

    return resolve(results)
#end def
add_unit_test(test_linesearchiteration_class)

