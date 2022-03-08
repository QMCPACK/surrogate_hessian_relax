#!/usr/bin/env python3

from numpy import array

verbose = True
unit_tests = []
integration_tests = []

def add_unit_test(func):
    unit_tests.append(func)
#end def

def add_integration_test(func):
    integration_tests.append(func)
#end def

def run_one_test(func, t_this, t_max, R = None):
    name = func.__name__
    if R is not None:
        if R not in name:
            return
        #end if
    #end if
    res = func()
    status = '    OK' if res else 'FAILED'
    print('  {:>4d}/{:<4d} {:40s} {:6s}'.format(t_this + 1, t_max, name[:40], status))
#end def

def run_tests(tests, R = None):
    t_max = len(tests)
    for t_this, test in enumerate(tests):
        run_one_test(test, t_this, t_max, R = R)
    #end for
#end def

def run_all_tests(**kwargs):
    run_unit_tests(**kwargs)
    run_integration_tests(**kwargs)
#end def

def run_unit_tests(R = None):
    print('Deterministic unit tests:')
    run_tests(unit_tests, R = R)
#end def

def run_integration_tests(R = None):
    print('Stochastic integration unit tests:')
    run_tests(integration_tests, R = R)
#end def


def match_values(val1, val2, tol = 1e-8):
    if abs(val1 - val2) < tol:
        return True
    else:
        print('{} and {} differ by {} > {}'.format(val1, val2, abs(val1 - val2), tol))
        return False
    #end if
#end def

def match_all_values(val1, val2, tol = 1e-8, expect_false = False):
    val1 = array(val1).flatten()
    val2 = array(val2).flatten()
    for v,val in enumerate(abs(val1 - val2)):
        if val > tol:
            if not expect_false:
                print('row {}: {} and {} differ by {} > {}'.format(v, val1, val2, abs(val1 - val2), tol))
            return False
        #end if
    #end for
    return True
#end def

def resolve(results):
    if verbose:
        if all(results):
            return True
        else:
            print(results)
            return False
        #end if
    #end if
#end def
