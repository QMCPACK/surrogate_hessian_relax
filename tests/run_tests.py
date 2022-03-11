#!/usr/bin/env python3

# add argument parser
import argparse
parser = argparse.ArgumentParser(description='Surrogate Hessian Relax unit tests')
parser.add_argument('-R', help = 'Filter test functions:\n  unit_* for deterministic tests\n  integration_* for stochastic integration tests')
parser.add_argument('-v', help = 'Verbose ouput for debugging:', default = False, action = 'store_true')
args = parser.parse_args()

if __name__=='__main__':
    import tools.unit
    import tools.integration
    import classes.unit
    from testing import run_all_tests
    run_all_tests(R = args.R, verbose = args.v)
#end if
