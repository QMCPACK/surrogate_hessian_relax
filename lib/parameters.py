#!/usr/bin/env python3

from numpy import linalg, pi, arccos, array, dot, isscalar, diag
from copy import deepcopy

from lib.util import match_to_tol


# distance between two atomic coordinates
def distance(r0, r1):
    r = linalg.norm(r0 - r1)
    return r
#end def


# bond angle between r0-rc and r1-rc bonds
def bond_angle(r0, rc, r1, units = 'ang'):
    v1 = r0 - rc
    v2 = r1 - rc
    cosang = dot(v1, v2) / linalg.norm(v1) / linalg.norm(v2)
    ang = arccos(cosang) * 180 / pi if units == 'ang' else arccos(cosang)
    return ang
#end def


def mean_distances(pairs):
    rs = []
    for pair in pairs:
        rs.append(distance(pair[0], pair[1]))
    #end for
    return array(rs).mean()
#end def


class ParameterBase():
    """Base class for representing an optimizable parameter"""
    kind = None
    value = None
    label = None
    unit = None

    def __init__(
        self,
        value,
        kind = '',
        label = 'p',
        unit = '',
    ):
        self.value = value
        self.kind = kind
        self.label = label
        self.unit = unit
    #end def

    def __repr__(self):
        return '{:>10s}: {:<10f} {:6s} ({})'.format(self.label, self.value, self.unit, self.kind)
    #end def
#end class


class Parameter(ParameterBase):
    """Class for representing an optimizable parameter"""

    def __init__(
        self,
        value,
        kind = '',
        label = 'p',
        unit = '',
    ):
        ParameterBase.__init__(self, value, kind, label, unit)
    #end def
#end class


class ParameterSet():
    """Base class for representing a set of parameters to optimize"""
    params = None
    params_err = None
    value = None  # energy value
    error = None  # errorbar
    label = None  # label for identification

    def __init__(
        self,
        params = None,
        params_err = None,
        value = None,
        error = None,
        label = None,
    ):
        self.label = label
        if params is not None:
            self.set_params(params, params_err)
        #end if
        if value is not None:
            self.set_value(value, error)
        #end if
    #end def

    def set_params(self, params, params_err = None):
        self.params = array(params).flatten()
        self.params_err = params_err
        self.unset_value()
    #end def

    def set_value(self, value, error = None):
        assert self.params is not None, 'Cannot assign value to abstract structure, set params first'
        self.value = value
        self.error = error
    #end def

    def unset_value(self):
        self.value = None
        self.error = None
    #end def

    def shift_params(self, dparams):
        assert self.params is not None, 'params has not been set'
        self.params += dparams
        self.unset_value()
    #end def

    def copy(
        self,
        params = None,
        params_err = None,
        label = None,
        **kwargs,
    ):
        structure = deepcopy(self)
        if params is not None:
            structure.set_params(params, params_err)
        #end if
        if label is not None:
            structure.label = label
        #end if
        return structure
    #end def

    def check_consistency(self):
        return True
    #end def
#end class


class ParameterStructureBase(ParameterSet):
    """Base class for representing a mapping between reducible positions (pos, axes) and irreducible parameters."""
    forward_func = None  # mapping function from pos to params
    backward_func = None  # mapping function from params to pos
    pos = None  # real-space position
    axes = None  # cell axes
    dim = None  # dimensionality
    elem = None  # list of elements
    unit = None  # position units
    # TODO: parameter trust intervals
    # FLAGS
    periodic = False  # is the line-search periodic; will axes be supplied to forward_func
    irreducible = False  # is the parameter mapping irreducible
    consistent = False  # are forward and backward mappings consistent

    def __init__(
        self,
        forward = None,  # pos to params
        backward = None,  # params to pos
        pos = None,
        axes = None,
        elem = None,
        params = None,
        params_err = None,
        periodic = False,
        value = None,
        error = None,
        label = None,
        unit = None,
        dim = 3,
        translate = True,  # attempt to translate pos
        **kwargs,
    ):
        self.dim = dim
        self.periodic = periodic
        self.label = label
        self.set_forward(forward)
        self.set_backward(backward)
        if pos is not None:
            self.set_position(pos, translate = translate)
        #end if
        if axes is not None:
            self.set_axes(axes)
        #end if
        if params is not None:
            self.set_params(params, params_err)
        #end if
        if value is not None:
            self.set_value(value, error)
        #end if
        if elem is not None:
            self.set_elem(elem)
        #end if
    #end def

    def set_forward(self, forward):
        self.forward_func = forward
        if self.pos is not None:
            self.forward()
        #end if
        self.check_consistency()
    #end def

    def set_backward(self, backward):
        self.backward_func = backward
        if self.params is not None:
            self.backward()
        #end if
        self.check_consistency()
    #end def

    # similar but alternative to Nexus function set_pos()
    def set_position(self, pos, translate = True):
        pos = array(pos)
        assert pos.size % self.dim == 0, 'Position vector inconsistent with {} dimensions!'.format(self.dim)
        self.pos = array(pos).reshape(-1, self.dim)
        if self.forward_func is not None:
            self.forward()
            if translate and self.backward_func is not None:
                self.backward()
            #end if
        #end if
        self.unset_value()  # setting pos will unset value
        self.check_consistency()
    #end def

    def set_axes(self, axes, check = True):
        if array(axes).size == self.dim:
            axes = diag(axes)
        else:
            axes = array(axes)
            assert axes.size == self.dim**2, 'Axes vector inconsistent with {} dimensions!'.format(self.dim)
            axes = array(axes).reshape(self.dim, self.dim)
        #end if
        try:
            self.reset_axes(axes)  # use nexus method to get kaxes
        except AttributeError:
            self.axes = axes
        #end try
        self.unset_value()  # setting axes will unset value
        if check:
            self.check_consistency()
        #end if
    #end def

    def set_elem(self, elem):
        self.elem = array(elem)
    #end def 

    def set_params(self, params, params_err = None):
        self.params = array(params).flatten()
        self.params_err = params_err
        if self.backward_func is not None:
            self.backward()
        #end if
        self.unset_value()
        self.check_consistency()
    #end def

    def set_value(self, value, error = None):
        assert self.params is not None or self.pos is not None, 'Cannot assign value to abstract structure, set params or pos first'
        self.value = value
        self.error = error
    #end def

    def unset_value(self):
        self.value = None
        self.error = None
    #end def

    def forward(self, pos = None, axes = None):
        """Propagate current structure from pos, axes (current, unless provided) to params"""
        assert self.forward_func is not None, 'Forward mapping has not been supplied'
        if pos is None:
            assert self.pos is not None, 'Must supply position for forward mapping'
            pos = self.pos
            axes = self.axes
        else:
            self.pos = pos
            self.axes = axes
        #end if
        assert not self.periodic or self.axes is not None, 'Must supply axes for periodic forward mappings'
        self.params = self._forward(pos, axes)
    #end def

    def _forward(self, pos, axes = None):
        """Perform forward mapping: return new params"""
        if self.periodic:
            return array(self.forward_func(array(pos), axes))
        else:
            return array(self.forward_func(array(pos)))
        #end if
    #end def

    def backward(self, params = None):
        """Propagate current structure from params (current, unless provided) to pos, axes"""
        assert self.backward_func is not None, 'Backward mapping has not been supplied'
        if params is None:
            assert self.params is not None, 'Must supply params for backward mapping'
            params = self.params
        else:
            self.params = params
        #end if
        self.pos, axes = self._backward(params)
        if self.periodic:
            self.set_axes(axes, check = False)  # enable checkups but avoid recursion loop
        #end if
    #end def

    def _backward(self, params):
        """Perform backward mapping: return new pos, axes"""
        if self.periodic:
            pos, axes = self.backward_func(array(params))
            return array(pos).reshape(-1, 3), array(axes).reshape(-1, 3)
        else:
            return array(self.backward_func(array(params))).reshape(-1, 3), None
        #end if
    #end def

    def check_consistency(
        self,
        params = None,
        pos = None,
        axes = None,
        tol = 1e-7,
        verbose = False,
    ):
        """Check consistency of the current or provided set of positions and params."""
        if pos is None and params is None:
            # if neither params nor pos are given, check and store internal consistency
            if self.pos is None and self.params is None:
                # without either set of coordinates the mapping is inconsistent
                consistent = False
            else:
                consistent = self._check_consistency(self.params, self.pos, self.axes, tol)
            #end if
            self.consistent = consistent
            if consistent:
                self.forward()
                self.backward()
            #end if
        else:
            consistent = self._check_consistency(params, pos, axes, tol)
        #end if
        return consistent
    #end def

    def _check_consistency(self, params, pos, axes, tol = 1e-7):
        """Check consistency of present forward-backward mapping.
        If params or pos/axes are supplied, check at the corresponding points. If not, check at the present point.
        """
        if self.forward_func is None or self.backward_func is None:
            return False
        #end if
        if pos is None and params is None:
            return False
        elif pos is not None and params is not None:
            # if both params and pos are given, check their internal consistency
            pos = array(pos)
            params = array(params)
            axes = array(axes)
            params_new = self._forward(pos, axes)
            pos_new, axes_new = self._backward(params)
            if self.periodic:
                return match_to_tol(params, params_new, tol) and match_to_tol(pos, pos_new, tol) and match_to_tol(axes, axes_new, tol)
            else:
                return match_to_tol(params, params_new, tol) and match_to_tol(pos, pos_new, tol)
            #end if
        elif params is not None:
            return self._check_params_consistency(array(params), tol)
        else:  # pos is not None
            return self._check_pos_consistency(array(pos), array(axes), tol)
        #end if
    #end def

    def _check_pos_consistency(self, pos, axes, tol = 1e-7):
        if self.periodic:
            params = self._forward(pos, axes)
            pos_new, axes_new = self._backward(params)
            consistent = match_to_tol(pos, pos_new, tol) and match_to_tol(axes, axes_new, tol)
        else:
            params = self._forward(pos, axes)
            pos_new, axes_new = self._backward(params)
            consistent = match_to_tol(pos, pos_new, tol)
        #end if
        return consistent
    #end def

    def _check_params_consistency(self, params, tol = 1e-7):
        if self.periodic:
            pos, axes = self._backward(params)
            params_new = self._forward(pos, axes)
        else:
            pos, axes = self._backward(params)
            params_new = self._forward(pos, axes)
        #end if
        return match_to_tol(params, params_new, tol)
    #end def

    def _shift_pos(self, dpos):
        if isscalar(dpos):
            return self.pos + dpos
        #end if
        dpos = array(dpos)
        assert self.pos.size == dpos.size
        return (self.pos.flatten() + dpos.flatten()).reshape(-1, self.dim)
    #end def

    def shift_pos(self, dpos):
        assert self.pos is not None, 'position has not been set'
        self.pos = self._shift_pos(dpos)
        self.forward()
        self.check_consistency()
    #end def

    def _shift_params(self, dparams):
        if isscalar(dparams):
            return self.params + dparams
        #end if
        dparams = array(dparams)
        assert self.params.size == dparams.size
        return self.params + dparams
    #end def

    def shift_params(self, dparams):
        assert self.params is not None, 'params has not been set'
        self.params = self._shift_params(dparams)
        self.backward()
        self.check_consistency()
    #end def

    def copy(
        self,
        params = None,
        params_err = None,
        label = None,
        pos = None,
        axes = None,
        **kwargs,
    ):
        structure = deepcopy(self)
        if params is not None:
            structure.set_params(params, params_err)
        #end if
        if pos is not None:
            structure.set_pos(pos)
        #end if
        if axes is not None:
            structure.set_axes(axes)
        #end if
        if label is not None:
            structure.label = label
        #end if
        return structure
    #end def

    def pos_difference(self, pos_ref):
        dpos = pos_ref.reshape(-1, 3) - self.pos
        return dpos
    #end def

    def jacobian(self, dp = 0.001):
        assert self.consistent, 'The mapping must be consistent'
        jacobian = []
        for p in range(len(self.params)):
            params_this = self.params.copy()
            params_this[p] += dp
            pos, axes = self._backward(params_this)
            dpos = self.pos_difference(pos)
            jacobian.append(dpos.flatten() / dp)
        #end for
        return array(jacobian).T
    #end def

    def __repr__(self):
        string = self.__class__.__name__
        if self.label is not None:
            string += ' ({})'.format(self.label)
        #end if
        if self.forward_func is None:
            string += '\n  forward mapping: not set'
        else:
            string += '\n  forward mapping: {}'.format(self.forward_func)
        #end if
        if self.backward_func is None:
            string += '\n  backward mapping: not set'
        else:
            string += '\n  backward mapping: {}'.format(self.backward_func)
        #end if
        if self.consistent:
            string += '\n  consistent: yes'
        else:
            string += '\n  consistent: no'
        #end if
        # params
        if self.params is None:
            string += '\n  params: not set'
        else:
            string += '\n  params:'
            for param in self.params:
                string += '\n    {:<10f}'.format(param)  # TODO: ParameterBase
            #end for
        #end if
        # pos
        if self.pos is None:
            string += '\n  pos: not set'
        else:
            string += '\n  pos ({:d} atoms)'.format(len(self.pos))
            for elem, pos in zip(self.elem, self.pos):
                string += '\n    {:2s} {:<6f} {:<6f} {:<6f}'.format(elem, pos[0], pos[1], pos[2])
            #end for
        #end if
        if self.periodic:
            string += '\n  periodic: yes'
            if self.axes is None:
                string += '\n  axes: not set'
            else:
                string += '\n  axes:'
                for axes in self.axes:
                    string += '\n    {:<6f} {:<6f} {:<6f}'.format(axes[0], axes[1], axes[2])
                #end for
            #end if
        else:
            string += '\n  periodic: no'
        #end if
        return string
    #end def

#end class


# Class for physical structure (Nexus)
try:
    from structure import Structure

    class ParameterStructure(ParameterStructureBase, Structure):
        kind = 'nexus'

        def __init__(
            self,
            forward = None,
            backward = None,
            params = None,
            **kwargs
        ):
            ParameterStructureBase.__init__(
                self,
                forward = forward,
                backward = backward,
                params = params,
                **kwargs,
            )
            self.to_nexus_structure(**kwargs)
        #end def

        def to_nexus_structure(
            self,
            kshift = (0, 0, 0),
            kgrid = (1, 1, 1),
            units = 'A',
            **kwargs
        ):
            s_args = {
                'elem': self.elem,
                'pos': self.pos,
                'units': units,
            }
            if self.axes is not None:
                s_args.update({
                    'axes': self.axes,
                    'kshift': kshift,
                    'kgrid': kgrid,
                })
            #end if
            Structure.__init__(self, **s_args)
        #end def

        def to_nexus_only(self):
            self.forward_func = None
            self.backward_func = None
        #end def

    #end class
except ModuleNotFoundError:  # plain implementation if nexus not present
    class ParameterStructure(ParameterStructureBase):
        kind = 'plain'
    #end class
#end try
