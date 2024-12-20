#!/usr/bin/env python

from numpy import array
from pytest import raises
from stalk.util import match_to_tol

from ..assets.h2o import pos_H2O, elem_H2O, forward_H2O, backward_H2O
from ..assets.h2 import pos_H2, forward_H2, backward_H2
from ..assets.gese import params_GeSe, forward_GeSe, backward_GeSe, elem_GeSe

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


def test_parameter_tools():
    from stalk.params import distance, bond_angle, mean_distances
    # water molecule
    pos = array('''
    0.00000        0.00000        0.11779
    0.00000        0.75545       -0.47116
    0.00000       -0.75545       -0.47116
    '''.split(), dtype=float).reshape(-1, 3)
    assert match_to_tol(distance(pos[0], pos[1]), 0.957897074324)
    assert match_to_tol(distance(pos[0], pos[2]), 0.957897074324)
    assert match_to_tol(distance(pos[1], pos[2]), 1.5109)
    assert match_to_tol(mean_distances(
        [(pos[0], pos[1]), (pos[0], pos[2])]), 0.957897074324)
    assert match_to_tol(bond_angle(pos[1], pos[0], pos[2]), 104.1199307245)
# end def


# Test Parameter class
def test_parameter_class():
    # TODO: add features, add meaningful tests
    from stalk.params import Parameter

    # Cannot construct without a value
    with raises(TypeError):
        Parameter()
    # end with
    # test empty
    p = Parameter(0.0)
    assert p.value == 0.0
    assert p.error == 0.0
    assert p.kind == ''
    assert p.unit == ''
    assert p.label == 'p'

    # test nominal
    p = Parameter(0.74, 0.01, 'kind', 'test', 'unit')
    assert p.value == 0.74
    assert p.error == 0.01
    assert p.kind == 'kind'
    assert p.label == 'test'
    assert p.unit == 'unit'
# end def


# Test ParameterSet class
def test_parameterset_class():
    from stalk import ParameterSet

    # test empty
    s = ParameterSet()
    assert s.p_list == []
    assert s.value is None
    assert s.error is None
    assert s.label is None
# end def


# Test ParameterStructureBase class
def test_parameterstructurebase_class():
    from stalk.params import ParameterStructureBase

    s = ParameterStructureBase()
    # test H2 (open; 1 parameter)

    # test inconsistent pos vector
    with raises(AssertionError):
        s.set_position([0.0, 0.0])
    # end with
    # test premature backward mapping
    assert s.map_backward()[0] is None
    assert s.map_backward()[1] is None
    # cannot be consistent without mapping functions
    assert not s.check_consistency()

    # test backward mapping
    s.set_params([1.4])
    s.set_backward_func(backward_H2)  # pos should now be computed automatically
    assert match_to_tol(s.pos, pos_H2, tol=1e-5)

    assert not s.check_consistency()  # still not consistent, without forward mapping
    # test premature forward mapping
    assert s.map_forward() is None

    s.set_forward_func(forward_H2)  # set forward mapping
    assert match_to_tol(s.pos, [0.0, 0.0, 0.7, 0.0, 0.0, -0.7], tol=1e-5)
    # set another pos
    s.set_position([0.0, 0.0, 0.0, 0.0, 0.0, 1.6])
    # params computed automatically
    assert match_to_tol(s.params, 1.6, tol=1e-5)
    assert s.check_consistency()  # finally consistent
    # also consistent at another point
    assert s.check_consistency(params=[1.3])
    assert s.check_consistency(pos=pos_H2)  # also consistent at another point
    # consistent set of arguments
    assert s.check_consistency(pos=pos_H2 * 0.5, params=[0.7])
    # inconsistent set of arguments
    assert not s.check_consistency(pos=pos_H2 * 0.5, params=[1.4])

    # test H2O (open; 2 parameters)
    s = ParameterStructureBase(pos=pos_H2O, forward=forward_H2O, elem=elem_H2O)
    params_ref = [0.95789707432, 104.119930724]
    assert match_to_tol(s.params, params_ref, tol=1e-5)

    # add backward mapping
    s.set_backward_func(backward_H2O)
    pos_ref = [[0.,          0.,          0.],
               [0.,          0.75545,     0.58895],
               [0.,         -0.75545,     0.58895]]
    assert match_to_tol(s.pos, pos_ref, tol=1e-5)
    assert s.check_consistency()

    # test another set of parameters
    s.set_params([1.0, 120.0])
    pos2_ref = [[0.,          0.,          0.],
                [0.,          0.8660254,   0.5],
                [0.,         -0.8660254,   0.5]]
    assert match_to_tol(s.params, [1.0, 120.0], tol=1e-5)
    assert match_to_tol(s.pos, pos2_ref, tol=1e-5)

    jac_ref = array('''
    0.          0.
    0.          0.
    0.          0.
    0.          0.
    0.8660254   0.00436329
    0.5        -0.00755752
    0.          0.
   -0.8660254  -0.00436329
    0.5        -0.00755752
    '''.split(), dtype=float).reshape(-1, 2)
    assert match_to_tol(jac_ref, s.jacobian())

    # test periodic structure
    s = ParameterStructureBase(
        forward_GeSe,  # forward
        backward_GeSe,  # backward
        None,  # pos
        None,  # axes
        elem_GeSe,  # elem
        params_GeSe,
        None,  # params_err
        value=-10.0,  # value
        error=0.1,  # error
        label='GeSe test',  # label
        unit='crystal',  # unit
        dim=3,  # dim
    )
    pos_orig = s.pos
    s.shift_params([0.1, 0.1, 0.0, -0.1, 0.05])
    params_ref = [4.360000, 4.050000, 0.414000, 0.456000, 0.610000]
    pos_ref = array('''
    0.414000 0.250000 0.456000
    0.914000 0.750000 0.544000
    0.500000 0.250000 0.390000
    0.000000 0.750000 0.610000
    '''.split(), dtype=float)
    axes_ref = array('''
    4.360000 0.000000 0.000000
    0.000000 4.050000 0.000000
    0.000000 0.000000 20.000000
    '''.split(), dtype=float)
    assert match_to_tol(s.params, params_ref)
    assert match_to_tol(s.pos, pos_ref)
    assert match_to_tol(s.axes, axes_ref)
    dpos = pos_orig.flatten() - pos_ref
    s.shift_pos(dpos)
    params_ref2 = [4.360000, 4.050000, 0.414000, 0.556000, 0.560000]
    assert match_to_tol(s.params, params_ref2)
    assert match_to_tol(s.pos, pos_orig)
# end def


# Test ParameterStructure class
def test_parameterstructure_class():
    from stalk import ParameterStructure
    # test empty
    s = ParameterStructure()
    if s.kind == 'nexus':  # special tests for nexus Structure
        pass  # TODO
    # end if

    # test copy
    s1 = ParameterStructure(
        pos=pos_H2O, forward=forward_H2O, label='1', elem=elem_H2O)
    s2 = s1.copy(params=s1.params * 1.5, label='2')
    assert not match_to_tol(s1.params, s2.params)
    assert s1.label == '1'
    assert s2.label == '2'
    assert s1.forward_func == s2.forward_func
# end def
