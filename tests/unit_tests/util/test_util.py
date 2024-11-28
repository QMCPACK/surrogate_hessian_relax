#!/usr/bin/env python

from numpy import array, ones, random, linspace, polyval
from pytest import raises

from stalk.util import match_to_tol, get_min_params, get_fraction_error

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


def test_units():
    from stalk.util.util import Bohr, Ry, Hartree
    match_to_tol(Bohr, 0.5291772105638411, 1e-10)
    match_to_tol(Ry, 13.605693012183622, 1e-10)
    match_to_tol(Hartree, 2 * Ry, 1e-10)
# end def


def test_get_min_params():

    # Test 0-dim (degraded)
    with raises(AssertionError):
        get_min_params([], [], pfn=0)
    # end with

    # Test 1-dim (degraded)
    with raises(AssertionError):
        get_min_params([], [], pfn=1)
    # end with

    # Test 2-dim (degraded inputs)
    with raises(AssertionError):
        get_min_params([0.0], [], pfn=2)
    # end with
    with raises(AssertionError):
        get_min_params([], [0.0], pfn=2)
    # end with
    with raises(AssertionError):
        get_min_params([0.0, 0.0], [0.0, 0.0], pfn=2)
    # end with

    # Test 2-dim (nominal)
    p2_ref = array([1.23, 2.34, 3.45])
    x_in = linspace(-4, 4, 3)
    y_in = polyval(p2_ref, x_in)
    xmin2, ymin2, pf2 = get_min_params(x_in, y_in, pfn=2)
    # Should return the same pf, compare results to analytical solutions
    assert match_to_tol(xmin2, -p2_ref[1] / 2 / p2_ref[0])  # x* = -b/2a
    assert match_to_tol(ymin2, -p2_ref[1]**2 / 4 / p2_ref[0] + p2_ref[2])  # y* = -b^2/4a
    assert match_to_tol(pf2, p2_ref)

    # Test 3-dim (nominal)
    p3_ref = array([3.0, 0.0, 0.0, 50.0])
    x_in = linspace(-4, 4, 4)
    y_in = polyval(p3_ref, x_in)
    xmin3, ymin3, pf3 = get_min_params(x_in, y_in, pfn=3)
    assert match_to_tol(xmin3, 0.0, 1e-6)  # x* = 0
    assert match_to_tol(ymin3, 50.0)  # y* = -b^2/4a
    assert match_to_tol(pf3, p3_ref)  

# end def


def test_get_fraction_error():

    # Test data
    N = array(range(101)) + 50
    N_skew = array(range(101)) + 50
    # Making one side '99's does not affect the median nor the errorbar
    N_skew[0:50] = 99 * ones(50)
    random.shuffle(N)
    random.shuffle(N_skew)

    # Test degraded
    with raises(ValueError):
        err, ave_b = get_fraction_error(N, fraction=0.5)
    # end with
    with raises(ValueError):
        err, ave_b = get_fraction_error(N, fraction=-1e-10)
    # end with

    # Test nominal
    for frac, target in zip([0.0, 0.01, 0.1, 0.49], [50, 49, 40, 1]):
        ave_b, err_b = get_fraction_error(N_skew, fraction=frac, both=True)
        ave, err = get_fraction_error(N_skew, fraction=frac, both=False)
        assert match_to_tol(err, target)
        assert match_to_tol(ave, 100)
        assert match_to_tol(err_b[0], 1)
        assert match_to_tol(err_b[1], target)
        assert match_to_tol(ave_b, 100)
    # end for

# end def


def test_match_to_tol():
    assert match_to_tol([[0.0, -0.1]], [0.1, -0.2], 0.1 + 1e-8)
# end def
