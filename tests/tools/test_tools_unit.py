#!/usr/bin/env python

from numpy import array, exp, nan, isnan, random
from pytest import raises

from testing import add_unit_test, match_values


def test_parameter_tools():
    from surrogate_tools import distance,bond_angle,mean_distances
    # water molecule
    pos = array('''
    0.00000        0.00000        0.11779
    0.00000        0.75545       -0.47116
    0.00000       -0.75545       -0.47116
    '''.split(),dtype=float).reshape(-1,3)
    assert match_values(distance(pos[0],pos[1]),0.957897074324)
    assert match_values(distance(pos[0],pos[2]),0.957897074324)
    assert match_values(distance(pos[1],pos[2]),1.5109)
    assert match_values(mean_distances([(pos[0],pos[1]),(pos[0],pos[2])]),0.957897074324)
    assert match_values(bond_angle(pos[1],pos[0],pos[2]),104.1199307245)
#end def
add_unit_test(test_parameter_tools)


def test_get_min_params():
    from surrogate_tools import get_min_params
    # data based on morse potential
    morse = lambda p,r: p[2]*((1-exp(-(r-p[0])/p[1]))**2-1)+p[3]
    p = array([3.0,1.5,1.0,0.0])
    x_in = array([1,2,3,4,5,6,7])
    y_in =  morse(p,x_in)
    
    ymin1,xmin1,pf1 = get_min_params(x_in,y_in,pfn=1) # this should fail
    ymin2,xmin2,pf2 = get_min_params(x_in,y_in,pfn=2)
    ymin3,xmin3,pf3 = get_min_params(x_in,y_in,pfn=3)
    ymin4,xmin4,pf4 = get_min_params(x_in,y_in,pfn=4)
    ymin5,xmin5,pf5 = get_min_params(x_in,y_in,pfn=5)
    ymin6,xmin6,pf6 = get_min_params(x_in,y_in,pfn=6)

    assert isnan(xmin1)
    assert match_values(xmin2,4.756835779652)
    assert match_values(xmin3,3.467303536188)
    assert match_values(xmin4,2.970625073283)
    assert match_values(xmin5,2.887435491013)
    assert match_values(xmin6,2.965039959160)
    assert isnan(ymin1)
    assert match_values(ymin2,-1.6348441686)
    assert match_values(ymin3,-1.5560903945)
    assert match_values(ymin4,-1.2521783062)
    assert match_values(ymin5,-1.0431982684)
    assert match_values(ymin6,-1.0005432843)

    # TODO: test faulty behavior
#end def
add_unit_test(test_get_min_params)


def test_get_fraction_error():
    from surrogate_tools import get_fraction_error
    N = array(range(101))+50
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
