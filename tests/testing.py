#!/usr/bin/env python3

unit_tests = []
integration_tests = []

def add_unit_test(func):
    unit_tests.append(func)
#end def

def add_integration_test(func):
    integration_tests.append(func)
#end def

def run_one_test(func, t_this, t_max):
    res = func()
    name = func.__name__
    status = 'PASSED' if res else 'FAILED'
    print('  {}/{} {} {}'.format(t_this + 1, t_max, name, status))
#end def

def run_tests(tests):
    t_max = len(tests)
    for t_this, test in enumerate(tests):
        run_one_test(test, t_this, t_max)
    #end for
#end def

def run_unit_tests():
    run_tests(unit_tests)
#end def

def run_integration_tests():
    run_tests(integration_tests)
#end def

def match_values(val1, val2, tol = 1e-10):
    if abs(val1 - val2) < tol:
        return True
    else:
        print('{} and {} differ by {} > {}'.format(val1, val2, abs(val1 - val2), tol))
        return False
    #end if
#end def
