#!/usr/bin/env python

from numpy import array, exp, random
from pytest import raises

from testing import add_unit_test, match_values


def test_get_min_params():
    from surrogate_classes import get_min_params
    # data based on morse potential
    morse = lambda p,r: p[2]*((1-exp(-(r-p[0])/p[1]))**2-1)+p[3]
    p = array([3.0,1.5,1.0,0.0])
    x_in = array([1,2,3,4,5,6,7])
    y_in =  morse(p,x_in)
   
    with raises(AssertionError): 
        xmin1,ymin1,pf1 = get_min_params(x_in,y_in,pfn=1) # this should fail
    #end with
    xmin2,ymin2,pf2 = get_min_params(x_in,y_in,pfn=2)
    xmin3,ymin3,pf3 = get_min_params(x_in,y_in,pfn=3)
    xmin4,ymin4,pf4 = get_min_params(x_in,y_in,pfn=4)
    xmin5,ymin5,pf5 = get_min_params(x_in,y_in,pfn=5)
    xmin6,ymin6,pf6 = get_min_params(x_in,y_in,pfn=6)

    assert match_values(xmin2,4.756835779652)
    assert match_values(xmin3,3.467303536188)
    assert match_values(xmin4,2.970625073283)
    assert match_values(xmin5,2.887435491013)
    assert match_values(xmin6,2.965039959160)
    assert match_values(ymin2,-1.6348441686)
    assert match_values(ymin3,-1.5560903945)
    assert match_values(ymin4,-1.2521783062)
    assert match_values(ymin5,-1.0431982684)
    assert match_values(ymin6,-1.0005432843)

    # TODO: test faulty behavior
#end def
add_unit_test(test_get_min_params)


def test_get_fraction_error():
    from surrogate_classes import get_fraction_error
    N = array(range(101)) + 50
    random.shuffle(N)
    for frac,target in zip([0.0,0.01,0.1,0.5],[50,49,40,0]):
        ave,err = get_fraction_error(N,fraction=frac,both=True)
        assert match_values(err[0], target)
        assert match_values(err[1], target)
        assert match_values(ave, 100)
    #end for
    # test too large fraction
    with raises(ValueError):
        err,ave = get_fraction_error(N,fraction=0.6)
    #end with 
    with raises(ValueError):
        err,ave = get_fraction_error(N,fraction=-0.1)
    #end with 
#end def
add_unit_test(test_get_fraction_error)
