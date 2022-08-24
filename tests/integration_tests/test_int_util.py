#!/usr/bin/env python

from numpy import array,exp,nan,isnan,random

from testing import match_values, add_integration_test


def int_get_fraction_error():
    from surrogate_tools import get_fraction_error
    # test normal distribution
    sigma = 1.0
    mu = 5.0
    N = sigma*random.randn(1000) + mu
    for frac,target in zip([0.025, 0.163, 0.5],[2 * sigma, sigma, 0.0]):
        ave,err = get_fraction_error(N, fraction = frac)
        assert match_values(err, target, tol = 1e-1)
        assert match_values(ave, mu, tol = 1e-1)
    #end for
    # TODO: test other distributions
#end def
add_integration_test(int_get_fraction_error)
