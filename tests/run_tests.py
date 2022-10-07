#!/usr/bin/env python3

# add argument parser
import argparse
parser = argparse.ArgumentParser(description='Surrogate Hessian Relax unit tests')
parser.add_argument('-R', help = 'Filter test functions:\n  unit_* for deterministic tests\n  integration_* for stochastic integration tests')
parser.add_argument('-v', help = 'Verbose ouput for debugging:', default = False, action = 'store_true')
parser.add_argument('-p', help = 'Pass AssertionError', default = False, action = 'store_true')
args = parser.parse_args()

if __name__=='__main__':
    from unit_tests import test_util
    from unit_tests import test_parameters
    from unit_tests import test_hessian
    from unit_tests import test_linesearch
    from unit_tests import test_targetlinesearch
    from unit_tests import test_targetparallellinesearch
    from unit_tests import test_parallellinesearch
    from unit_tests import test_linesearchiteration
    from integration_tests import test_int_util

    from testing import run_all_tests
    run_all_tests(R = args.R, verbose = args.v, pass_error = args.p)
#end if
