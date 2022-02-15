#!/usr/bin/env python3

# add argument parser
import argparse
parser = argparse.ArgumentParser(description='Surrogate Hessian Relax unit tests')
parser.add_argument('--unit',  default=True,  action='store_true', help='Run unit tests [TRUE]',)
parser.add_argument('--integration', default=False, action='store_true', help='Run integration tests [FALSE]',)
args = parser.parse_args()

if __name__=='__main__':
    # unit tests
    if args.unit:
        print('Running unit tests')
        import tools.unit
        from testing import run_unit_tests
        run_unit_tests()
    #end if
    
    # integration tests
    if args.integration:
        print('\nRunning integration tests')
        import tools.integration
        from testing import run_integration_tests
        run_integration_tests()
    #end if
#end if
