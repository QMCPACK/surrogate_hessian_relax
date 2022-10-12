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


class Parameter():
    """Base class for representing an optimizable parameter"""
    kind = None
    value = None
    error = None
    label = None
    unit = None

    def __init__(
        self,
        value,
        error = 0.0,
        kind = '',
        label = 'p',
        unit = '',
    ):
        self.value = value
        self.error = error
        self.kind = kind
        self.label = label
        self.unit = unit
    #end def

    @property
    def param_err(self):
        return 0.0 if self.error is None else self.error
    #end def

    def print_value(self):
        if self.error is None:
            print('{:<8.6f}             '.format(self.value))
        else:
            print('{:<8.6f} +/- {:<8.6f}'.format(self.value, self.error))
        #end if
    #end def

    def __str__(self):
        return '{:>10s}: {:<10f} {:6s} ({})'.format(self.label, self.value, self.unit, self.kind)
    #end def
#end class


class ParameterSet():
    """Base class for representing a set of parameters to optimize"""
    p_list = []  # list of Parameter objects
    value = None  # energy value
    error = None  # errorbar
    label = None  # label for identification

    def __init__(
        self,
        params = None,
        value = None,
        error = None,
        label = None,
        **kwargs,  # params_err, units, kinds
    ):
        self.label = label
        if params is not None:
            self.init_params(params, **kwargs)
        #end if
        if value is not None:
            self.set_value(value, error)
        #end if
    #end def

    def init_params(self, params, params_err = None, units = None, labels = None, kinds = None, **kwargs):
        if params_err is None:
            params_err = len(params) * [params_err]
        else:
            assert len(params_err) == len(params)
        #end if
        if units is None or isinstance(units, str):
            units = len(params) * [units]
        else:
            assert len(units) == len(params)
        #end if
        if kinds is None or isinstance(kinds, str):
            kinds = len(params) * [kinds]
        else:
            assert len(kinds) == len(params)
        #end if
        if labels is None:
            labels = len(params) * [labels]
        else:
            assert len(labels) == len(labels)
        #end if
        p_list = []
        for p, (param, param_err, unit, label, kind) in enumerate(zip(params, params_err, units, labels, kinds)):
            lab = label if label is not None else 'p{}'.format(p)
            parameter = Parameter(param, param_err, unit = unit, label = lab, kind = kind)
            p_list.append(parameter)
        #end for
        self.p_list = p_list
    #end def

    def set_params(self, params, params_err = None):
        if self.params is None:
            self.init_params(params, params_err)
        #end if
        if params_err is None:
            params_err = len(params) * [0.0]
        #end if
        for sparam, param, param_err in zip(self.p_list, params, params_err):
            sparam.value = param
            sparam.error = param_err
        #end for
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

    @property
    def params(self):
        if self.p_list == []:
            return None
        else:
            return array([p.value for p in self.p_list])
        #end if
    #end def

    @property
    def params_err(self):
        if self.p_list == []:
            return None
        else:
            return array([p.param_err for p in self.p_list])
        #end if
    #end def

    def shift_params(self, dparams):
        assert not self.p_list == [], 'params has not been set'
        for p, d in zip(self.p_list, dparams):
            p.value += d
        #end for
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
        **kwargs,  # kinds, labels, units
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
            self.init_params(params, params_err, **kwargs)
            self.set_params(self.params)
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
            self._forward(self.pos)
        #end if
        self._check_consistency()
    #end def

    def set_backward(self, backward):
        self.backward_func = backward
        if self.params is not None:
            self._backward(self.params)
        #end if
        self._check_consistency()
    #end def

    # similar but alternative to Nexus function set_pos()
    def set_position(self, pos, translate = True):
        pos = array(pos)
        assert pos.size % self.dim == 0, 'Position vector inconsistent with {} dimensions!'.format(self.dim)
        self.pos = array(pos).reshape(-1, self.dim)
        if self.forward_func is not None:
            self._forward(pos)
            if translate and self.backward_func is not None:
                self._backward(self.params)
            #end if
        #end if
        self.unset_value()  # setting pos will unset value
        self._check_consistency()
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
            self._check_consistency()
        #end if
    #end def

    def set_elem(self, elem):
        self.elem = array(elem)
    #end def

    def set_params(self, params, params_err = None):
        ParameterSet.set_params(self, params, params_err)
        if self.backward_func is not None:
            self._backward(self.params)
        #end if
        self._check_consistency()
    #end def

    def set_value(self, value, error = None):
        assert not self.p_list == [] or self.pos is not None, 'Cannot assign value to abstract structure, set params or pos first'
        self.value = value
        self.error = error
    #end def

    def unset_value(self):
        self.value = None
        self.error = None
    #end def

    def _forward(self, pos, axes = None):
        """Propagate current structure from pos, axes (current, unless provided) to params"""
        self.set_params(self.forward(pos, axes))
    #end def

    def forward(self, pos = None, axes = None):
        """Perform forward mapping: return new params"""
        assert self.forward_func is not None, 'Forward mapping has not been supplied'
        pos = pos if pos is not None else self.pos
        if self.periodic:
            axes = axes if axes is not None else self.axes
            return array(self.forward_func(array(pos), axes))
        else:
            return array(self.forward_func(array(pos)))
        #end if
    #end def

    def _backward(self, params):
        """Propagate current structure from params (current, unless provided) to pos, axes"""
        self.pos, axes = self.backward(params)
        if self.periodic:
            self.set_axes(axes, check = False)  # enable checkups but avoid recursion loop
        #end if
    #end def

    def backward(self, params = None):
        """Perform backward mapping: return new pos, axes"""
        assert self.backward_func is not None, 'Backward mapping has not been supplied'
        params = params if params is not None else self.params
        if self.periodic:
            pos, axes = self.backward_func(array(params))
            return array(pos).reshape(-1, 3), array(axes).reshape(-1, 3)
        else:
            return array(self.backward_func(array(params))).reshape(-1, 3), None
        #end if
    #end def

    def _check_consistency(self):
        """Check and store consistency status of the present mappings."""
        self.consistent = self.check_consistency()
    #end def

    def check_consistency(self, params = None, pos = None, axes = None, tol = 1e-7):
        """Check consistency of present forward-backward mapping.
        If params or pos/axes are supplied, check at the corresponding points. If not, check at the present point.
        """
        if self.forward_func is None or self.backward_func is None:
            return False
        #end if
        axes = axes if axes is not None else self.axes
        if pos is None and params is not None:
            return self._check_params_consistency(array(params), tol)
        elif pos is not None and params is None:
            return self._check_pos_consistency(array(pos), array(axes), tol)
        #end if
        # if both params and pos are given, check their internal consistency
        pos = array(pos) if pos is not None else self.pos
        params = array(params) if params is not None else self.params
        if pos is not None and params is not None:
            params_new = array(self.forward(pos, axes))
            pos_new, axes_new = self.backward(params)
            if self.periodic:
                return match_to_tol(params, params_new, tol) and match_to_tol(pos, pos_new, tol) and match_to_tol(axes, axes_new, tol)
            else:
                return match_to_tol(params, params_new, tol) and match_to_tol(pos, pos_new, tol)
            #end if
        else:
            return False
        #end if
    #end def

    def _check_pos_consistency(self, pos, axes, tol = 1e-7):
        if self.periodic:
            params = self.forward(pos, axes)
            pos_new, axes_new = self.backward(params)
            consistent = match_to_tol(pos, pos_new, tol) and match_to_tol(axes, axes_new, tol)
        else:
            params = self.forward(pos, axes)
            pos_new, axes_new = self.backward(params)
            consistent = match_to_tol(pos, pos_new, tol)
        #end if
        return consistent
    #end def

    def _check_params_consistency(self, params, tol = 1e-7):
        pos, axes = self.backward(params)
        params_new = self.forward(pos, axes)
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
        self._forward(self.pos)
        self.check_consistency()
    #end def

    def shift_params(self, dparams):
        ParameterSet.shift_params(self, dparams)
        self._backward(self.params)
        self._check_consistency()
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
            structure.set_position(pos)
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
            pos, axes = self.backward(params_this)
            dpos = self.pos_difference(pos)
            jacobian.append(dpos.flatten() / dp)
        #end for
        return array(jacobian).T
    #end def

    def __str__(self):
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
