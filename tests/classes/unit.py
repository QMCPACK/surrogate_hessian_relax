#!/usr/bin/env python

from numpy import array,exp,nan,isnan,random, polyval, linspace

from testing import add_unit_test, match_values, match_all_values, resolve

from surrogate_classes import LineSearch, LineSearchHessian
from surrogate_classes import AbstractLineSearch, LineSearchStructure
from surrogate_classes import TargetLineSearch, ParameterMapping
from surrogate_classes import AbstractTargetLineSearch, ParallelLineSearch
from surrogate_classes import LineSearchIteration
from classes.assets import pos_H2O, elem_H2O, forward_H2O, backward_H2O, hessian_H2O, pes_H2O
from classes.assets import pos_H2, elem_H2, forward_H2, backward_H2, hessian_H2
from classes.assets import params_GeSe, forward_GeSe, backward_GeSe, hessian_GeSe
from classes.assets import morse

# Test ParameterMapping class
def test_parameter_mapping():
    results = []

    # test H2 (open; 1 parameter)
    pm = ParameterMapping()

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
    results.append(match_all_values(pm.pos, pos_H2, tol=1e-5))

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
    pm = ParameterMapping(pos = pos_H2O, forward = forward_H2O)
    params_ref = [0.95789707432, 104.119930724]
    results.append(match_all_values(pm.params, params_ref, tol=1e-5))

    # add backward mapping
    pm.set_backward(backward_H2O)
    pos_ref = [[ 0.,          0.,          0.     ],
               [ 0.,          0.75545,     0.58895],
               [ 0.,         -0.75545,     0.58895]]
    results.append(match_all_values(pm.pos, pos_ref, tol=1e-5))
    results.append(pm.check_consistency())

    # test another set of parameters
    pos2 = pm.backward([1.0, 120.0])
    pos2_ref = [[ 0.,          0.,          0. ],
                [ 0.,          0.8660254,   0.5],
                [ 0.,         -0.8660254,   0.5]]
    results.append(match_all_values(pos2, pos2_ref, tol = 1e-5))

    # test GeSe (periodic; 5 parameters)

    # test when only mappings are provided

    return resolve(results)
#end def
add_unit_test(test_parameter_mapping)

# Test LinesearchStructure
def test_linesearch_structure():
    results = []

    # test empty
    s = LineSearchStructure()
    if s.kind=='nexus':  # special tests for nexus Structure
        pass  # TODO
    #end if

    # test copy
    s1 = LineSearchStructure(pos = pos_H2O, forward = forward_H2O, label = '1')
    s2 = s1.copy(params = s1.params * 1.5, label = '2')
    results.append(not match_all_values(s1.params, s2.params, expect_false = True))
    results.append(s1.label == '1')
    results.append(s2.label == '2')
    results.append(s1.forward_func == s2.forward_func)

    return resolve(results)
#end def
add_unit_test(test_linesearch_structure)


def test_linesearch_hessian():
    results = []

    structure = LineSearchStructure(
        forward = forward_H2O,
        backward = backward_H2O,
        pos = pos_H2O)
    hessian = LineSearchHessian(hessian_H2O)
    Lambda = hessian.Lambda
    dire = hessian.get_directions()
    dir0 = hessian.get_directions(0)
    dir1 = hessian.get_directions(1)
    Lambda_ref = array([1.07015621, 0.42984379])
    dire_ref = array([[ 0.94362832,  0.33100694],
                      [-0.33100694,  0.94362832]])
    results.append(match_all_values(dire, dire_ref))
    results.append(match_all_values(dir0, dire_ref[0]))
    results.append(match_all_values(dir1, dire_ref[1]))
    results.append(match_all_values(Lambda, Lambda_ref))
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
add_unit_test(test_linesearch_hessian)



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
        fit_kind = 'pf',
        pfn = 3)
    x0 = ls.get_x0()
    y0 = ls.get_y0()
    x0_ref = [2.0000000000, 0.0]
    y0_ref = [0.3428571428, 0.0]
    fit_ref = [ 0.0,  4.28571429e-01, -1.71428571e+00,  2.05714286e+00]
    results.append(match_all_values(x0, x0_ref))
    results.append(match_all_values(y0, y0_ref))
    results.append(match_all_values(ls.fit, fit_ref))

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

    bias_x, bias_y, bias_tot = ls.evaluate_bias(grid, values)
    # TODO

    return resolve(results)
#end def
add_unit_test(test_abstracttargetlinesearch_class)


# test LineSearch
def test_linesearch_class():
    results = []
    structure = LineSearchStructure(
        forward = forward_H2O,
        backward = backward_H2O,
        pos = pos_H2O)
    hessian = LineSearchHessian(hessian_H2O)

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
    results.append(match_all_values(gridR_d0, gridR_d0_ref))
    results.append(match_all_values(gridR_d1, gridR_d1_ref))
    results.append(match_all_values(gridW_d0, gridW_d0_ref))
    results.append(match_all_values(gridW_d1, gridW_d1_ref))

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

    s = LineSearchStructure(forward = forward_H2, backward = backward_H2, pos = pos_H2)
    h = LineSearchHessian(hessian_H2)

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
    Rs = linspace(1e-2, 0.5, 21)
    tls0.set_target(grid, values, interpolate_kind = 'pchip')
    biases_x, biases_y, biases_tot = tls0.compute_bias_R(Rs, verbose = False)
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
    results.append(match_all_values(Rs, biases_ref[:,0], tol=1e-5))
    results.append(match_all_values(biases_x, biases_ref[:,1], tol=1e-5))
    results.append(match_all_values(biases_y, biases_ref[:,2], tol=1e-5))
    results.append(match_all_values(biases_tot, biases_ref[:,3], tol=1e-5))

    # bias_mix = 0.4
    tls4 = TargetLineSearch(
        structure = s,
        hessian = h,
        d = 0,
        R = 0.5,
        fit_kind = 'pf',
        pfn = 3,
        bias_x0 = 0.0,
        bias_y0 = -0.5,
        bias_mix = 0.4,
        target_grid = grid,
        target_values = values,
        interpolate_kind = 'cubic',
    )
    biases_x, biases_y, biases_tot = tls4.compute_bias_R(Rs, verbose = False)
    biases_ref = array('''
    0.010000   -0.000005  -0.000000  0.000005   
    0.034500   -0.000004  -0.000002  0.000005   
    0.059000   -0.000005  -0.000008  0.000008   
    0.083500   -0.000015  -0.000030  0.000027   
    0.108000   -0.000043  -0.000083  0.000076   
    0.132500   -0.000092  -0.000185  0.000166   
    0.157000   -0.000179  -0.000367  0.000325   
    0.181500   -0.000316  -0.000660  0.000580   
    0.206000   -0.000514  -0.001104  0.000956   
    0.230500   -0.000786  -0.001744  0.001484   
    0.255000   -0.001155  -0.002638  0.002210   
    0.279500   -0.001626  -0.003845  0.003163   
    0.304000   -0.002216  -0.005442  0.004393   
    0.328500   -0.002941  -0.007513  0.005946   
    0.353000   -0.003810  -0.010155  0.007872   
    0.377500   -0.004830  -0.013475  0.010220   
    0.402000   -0.006016  -0.017601  0.013056   
    0.426500   -0.007365  -0.022666  0.016432   
    0.451000   -0.008893  -0.028843  0.020431   
    0.475500   -0.010611  -0.036323  0.025140   
    0.500000   -0.012500  -0.045289  0.030616  
    '''.split(),dtype=float).reshape(-1,4)
    results.append(match_all_values(Rs, biases_ref[:,0], tol=1e-5))
    results.append(match_all_values(biases_x, biases_ref[:,1], tol=1e-5))
    results.append(match_all_values(biases_y, biases_ref[:,2], tol=1e-5))
    results.append(match_all_values(biases_tot, biases_ref[:,3], tol=1e-5))

    return resolve(results)
#end def
add_unit_test(test_targetlinesearch_class)


def test_parallel_linesearch():
    results = []

    s = LineSearchStructure(
        forward = forward_H2O,
        backward = backward_H2O,
        pos = pos_H2O)
    s.shift_params([0.2, 0.2])
    h = LineSearchHessian(hessian_H2O)
    pls = ParallelLineSearch(
        hessian = h,
        structure = s,
        M = 9,
        Lambda_frac = 0.1,
        noise_frac = 0.0)
    results.append(not pls.protected)
    results.append(not pls.generated)
    results.append(not pls.loaded)
    results.append(not pls.calculated)

    ls0 = pls.ls_list[0]
    ls1 = pls.ls_list[1]

    # test grid
    ls0_grid_ref = array('''-0.4472136 -0.3354102 -0.2236068 -0.1118034  0.         0.1118034
                     0.2236068  0.3354102  0.4472136'''.split(), dtype=float)
    ls1_grid_ref = array('''-0.4472136 -0.3354102 -0.2236068 -0.1118034  0.         0.1118034
                     0.2236068  0.3354102  0.4472136'''.split(), dtype=float)
    results.append(match_all_values(ls0.grid, ls0_grid_ref))
    results.append(match_all_values(ls1.grid, ls1_grid_ref))
    # test params
    params0 = [structure.params for structure in ls0.structure_list]
    params1 = [structure.params for structure in ls1.structure_list]
    params0_ref = array('''
    0.73589366 104.17189992
    0.84139451 104.20890762
    0.94689537 104.24591532
    1.05239622 104.28292302
    1.15789707 104.31993072
    1.26339793 104.35693843
    1.36889878 104.39394613
    1.47439963 104.43095383
    1.57990049 104.46796153
    '''.split(),dtype=float)
    params1_ref = array('''
    1.30592788 103.89792731
    1.26892018 104.00342816
    1.23191248 104.10892902
    1.19490478 104.21442987
    1.15789707 104.31993072
    1.12088937 104.42543158
    1.08388167 104.53093243
    1.04687397 104.63643328
    1.00986627 104.74193414
    '''.split(), dtype=float)
    results.append(match_all_values(params0, params0_ref))
    results.append(match_all_values(params1, params1_ref))
    # test PES
    values0 = [pes_H2O(params) for params in params0]
    values1 = [pes_H2O(params) for params in params1]
    values0_ref = [-0.3423932143374829, -0.4615345963938761, -0.4916987800787321, -0.4717361196238385, -0.4254689840332648, -0.36717986856519547, -0.30515030620422656, -0.24393300567151105, -0.18580254017072284]
    values1_ref = [-0.34983482594288307, -0.386065211742347, -0.4109440197138217, -0.4241925612150868, -0.4254689840332648, -0.4143567581898992, -0.39035121431271413, -0.3528438153855756, -0.3011037911107758]
    results.append(match_all_values(values0, values0_ref))
    results.append(match_all_values(values1, values1_ref))

    # test loading without function
    pls.load_results()
    results.append(not pls.loaded)
    # manually enter values
    pls.load_results(values=[values0, values1])
    results.append(pls.loaded)

    ls0_x0_ref = -0.194484057865, 0.0
    ls0_y0_ref = -0.488792489027, 0.0
    ls1_x0_ref = -0.043526903585, 0.0
    ls1_y0_ref = -0.426522069151, 0.0
    results.append(match_all_values(ls0.get_x0(), ls0_x0_ref))
    results.append(match_all_values(ls0.get_y0(), ls0_y0_ref))
    results.append(match_all_values(ls1.get_x0(), ls1_x0_ref))
    results.append(match_all_values(ls1.get_y0(), ls1_y0_ref))
  
    next_params_ref = [  0.98878412, 104.21448193]
    results.append(match_all_values(pls.get_next_params(), next_params_ref))

    return resolve(results)
#end def
add_unit_test(test_parallel_linesearch)



# test linesearch iteration

# defined here to make functions global for pickling
s = LineSearchStructure(
    forward = forward_H2O,
    backward = backward_H2O,
    pos = pos_H2O)
s.shift_params([0.2, -0.2])
h = LineSearchHessian(hessian_H2O)
params_ref = s.forward(pos_H2O)

def job_H2O_pes(pos, path, noise, **kwargs):
    p = s.forward(pos)
    value = pes_H2O(p)
    return [(path, value, noise)]
#end def
def analyze_H2O_pes(path, job_data = None, **kwargs):
    for row in job_data:
        if path == row[0]:
            return row[1], row[2]
        #end if
    #end for
    return None
#end def

def test_linesearch_iteration():
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
    results.append(match_all_values(lsi.pls_list[-1].structure.params, [  0.89725537, 104.12804938]))

    # second iteration
    job_data = lsi.generate_jobs()
    lsi.load_results(job_data = job_data)
    lsi.propagate(write = True)
    results.append(match_all_values(lsi.pls_list[-1].structure.params, [  0.93244294, 104.1720672 ]))
    
    # third iteration
    job_data = lsi.generate_jobs()
    lsi.load_results(job_data = job_data)
    lsi.propagate(write = False)
    results.append(match_all_values(lsi.pls_list[-1].structure.params, [  0.93703957, 104.20617541]))

    # start over and load until second iteration
    lsi = LineSearchIteration(path = test_dir)
    results.append(len(lsi.pls_list) == 2)

    lsi.propagate(write = False)
    lsi.generate_jobs()
    lsi.load_results(job_data = job_data)
    lsi.propagate(write = False)
    results.append(match_all_values(lsi.pls_list[-1].structure.params, [  0.93703957, 104.20617541]))

    # TODO: test starting from surrogate
    # TODO: test stochastic line-search

    # remove test directory
    from shutil import rmtree
    rmtree(test_dir)

    return resolve(results)
#end def
add_unit_test(test_linesearch_iteration)

