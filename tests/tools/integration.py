#!/usr/bin/env python

from numpy import array,exp,nan,isnan,random

from testing import add_integration_test,match_values


def int_get_fraction_error():
    results = []
    from surrogate_tools import get_fraction_error
    # test normal distribution
    sigma = 1.0
    mu = 5.0
    N = sigma*random.randn(1000)+mu
    for frac,target in zip([0.025,0.163,0.5],[2*sigma,sigma,0.0]):
        ave,err = get_fraction_error(N,fraction=frac)
        results.append(match_values(err,target,tol=1e-1))
        results.append(match_values(ave,mu,tol=1e-1))
    #end for
    # TODO: test other distributions
    return all(results)
#end def
add_integration_test(int_get_fraction_error)
