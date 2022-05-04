#!/usr/bin/env python3

import pickle
from os import makedirs
from numpy import array, diag, linalg, linspace, savetxt, roots, nan, isnan, mean
from numpy import random, argsort, isscalar, ndarray, polyfit, polyval
from numpy import insert, append, where, polyder, argmin, median, argmax
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import broyden1
from functools import partial
from textwrap import indent
from copy import deepcopy

Bohr = 0.5291772105638411  # A
Ry = 13.605693012183622  # eV
Hartree = 27.211386024367243  # eV


def match_to_tol(val1, val2, tol = 1e-10):
    """Match the values of two vectors. True if all match, False if not."""
    assert len(val1) == len(val2), 'lengths of val1 and val2 do not match' + str(val1) + str(val2)
    for v1, v2 in zip(val1.flatten(), val2.flatten()):  # TODO: maybe vectorize?
        if abs(v2 - v1) > tol:
            return False
        #end if
    #end for
    return True
#end def

# Important function to resolve the local minimum of a curve
def get_min_params(x_n, y_n, pfn = 3, **kwargs):
    pf = polyfit(x_n, y_n, pfn)
    r = roots(polyder(pf))
    x_mins  = list(r[r.imag == 0].real)
    if len(x_mins) > 0:
        y_mins = polyval(pf, array(x_mins))
        imin = argmin(y_mins)
        y0 = y_mins[imin]
        x0 = x_mins[imin]
    else:
        y0 = nan
        x0 = nan
    #end if
    return x0, y0, pf
#end def

# Map W to R, given H
def W_to_R(W, H):
    R = (2 * W / H)**0.5
    return R
#end def


# Map R to W, given H
def R_to_W(R, H):
    W = 0.5 * H * R**2
    return W
#end def


# Estimate conservative (maximum) uncertainty from a distribution based on a percentile fraction
def get_fraction_error(data, fraction, both = False):
    if fraction < 0.0 or fraction > 0.5:
        raise ValueError('Invalid fraction')
    #end if
    data   = array(data, dtype = float)
    data   = data[~isnan(data)]        # remove nan
    ave    = median(data)
    data   = data[data.argsort()] - ave  # sort and center
    pleft  = abs(data[int((len(data) - 1) * fraction)])
    pright = abs(data[int((len(data) - 1) * (1 - fraction))])
    if both:
        err = [pleft, pright]
    else:
        err = max(pleft, pright)
    #end if
    return ave, err
#end def



class StructuralParameter():
    """Class for representing a physical structural parameter"""
    kind = None
    value = None
    label = None
    unit = None

    def __init__(
        self,
        value,
        kind = 'bond',
        label = 'r',
        unit = 'A',
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


class ParameterStructureBase():
    """Base class for representing a mapping between reducible positions (pos, axes) and irreducible parameters."""
    forward_func = None  # mapping function from pos to params
    backward_func = None  # mapping function from params to pos
    pos = None  # real-space position
    axes = None  # cell axes
    params = None  # reduced parameters
    params_err = None  # store parameter uncertainties
    dim = None  # dimensionality
    elem = None  # list of elements
    unit = None  # position units
    value = None  # energy value
    error = None  # errorbar
    label = None  # label for identification
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

    def set_axes(self, axes):
        if array(axes).size == self.dim:
            self.axes = diag(axes)
        else:
            axes = array(axes)
            assert axes.size == self.dim**2, 'Axes vector inconsistent with {} dimensions!'.format(self.dim)
            self.axes = array(axes).reshape(self.dim, self.dim)
        #end if
        self.unset_value()  # setting aces will unset value
        self.check_consistency()
    #end def

    def set_elem(self, elem):
        self.elem = elem
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
        assert self.params is not None or self.pos is not None, 'Assigning value to abstract structure, set params or pos first'
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
        if self.periodic:
            self.pos, self.axes = self._backward(params)
        else:
            self.pos, axes = self._backward(params)  # no update
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
        label = None,
        pos = None,
        axes = None,
        **kwargs,
    ):
        structure = deepcopy(self)
        if params is not None:
            structure.set_params(params)
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
                string += '\n    {:<10f}'.format(param)  # TODO: StructuralParameter
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

    #end class
except ModuleNotFoundError:  # plain implementation if nexus not present
    class ParameterStructure(ParameterStructureBase):
        kind = 'plain'
    #end class
#end try


# Class for parameter Hessian matrix
class ParameterHessian():
    hessian = None  # always stored in (Ry/A)**2
    Lambda = None
    structure = None
    U = None
    P = None
    D = None
    hessian_set = False  # flag whether hessian is set (True) or just initialized (False)

    def __init__(
        self,
        hessian = None,
        structure = None,
        hessian_real = None,
        **kwargs,  # units etc
    ):
        if structure is not None:
            self.set_structure(structure)
            if hessian_real is not None:
                self.init_hessian_real(hessian_real)
            else:
                self.init_hessian_structure(structure)
            #end if
        #end if
        if hessian is not None:
            self.init_hessian_array(hessian, **kwargs)
        #end if
    #end def

    def set_structure(self, structure):
        assert isinstance(structure, ParameterStructure), 'Structure must be ParameterStructure object'
        self.structure = structure
    #end def

    def init_hessian_structure(self, structure):
        assert structure.isinstance(ParameterStructure), 'Provided argument is not ParameterStructure'
        assert structure.check_consistency(), 'Provided ParameterStructure is incomplete or inconsistent'
        hessian = diag(len(structure.params) * [1.0])
        self._set_hessian(hessian)
        self.hessian_set = False  # this is not an appropriate hessian
    #end def

    def init_hessian_real(self, hessian_real, structure = None):
        structure = structure if structure is not None else self.structure
        jacobian = structure.jacobian()
        hessian = jacobian.T @ hessian_real @ jacobian
        self._set_hessian(hessian)
    #end def

    def init_hessian_array(self, hessian, **kwargs):
        hessian = self._convert_hessian(array(hessian), **kwargs)
        self._set_hessian(hessian)
    #end def

    def update_hessian(
        self,
        hessian,
    ):
        hessian = self._convert_hessian(array(hessian))
        P, D = hessian.shape
        Lambda, U = linalg.eig(hessian)
        assert P == self.P, 'Parameter count P={} does not match initial {}'.format(P, self.P)
        assert D == self.D, 'Direction count D={} does not match initial {}'.format(D, self.D)
        self._set_hessian(hessian)        
    #end def

    def _set_hessian(self, hessian):
        # TODO: assertions
        if len(hessian) == 1:
            Lambda = array(hessian[0])
            U = array([1.0])
            P, D = 1, 1
        else:
            Lambda, U = linalg.eig(hessian)
            P, D = hessian.shape
        #end if
        self.hessian = array(hessian)
        self.P, self.D = P, D
        self.Lambda, self.U = Lambda, U
        self.hessian_set = True
    #end def

    def get_directions(self, d = None):
        if d is None:
            return self.U.T
        else:
            return self.U.T[d]
        #end if
    #end def

    def get_lambda(self, d= None):
        if d is None:
            return self.Lambda
        else:
            return self.Lambda[d]
        #end if
    #end def

    def _convert_hessian(
        self,
        hessian,
        x_unit = 'A',
        E_unit = 'Ry',
    ):
        if x_unit == 'B':
            hessian *= Bohr**2
        elif x_unit == 'A':
            hessian *= 1.0
        else:
            raise ValueError('E_unit {} not recognized'.format(E_unit))
        #end if
        if E_unit == 'Ha':
            hessian /= (Hartree / Ry)**2
        elif E_unit == 'eV':
            hessian /= Ry**2
        elif E_unit == 'Ry':
            hessian /= 1.0
        else:
            raise ValueError('E_unit {} not recognized'.format(E_unit))
        #end if
        return hessian
    #end def

    def get_hessian(self, **kwargs):
        return self._convert_hessian(self.hessian, **kwargs)
    #end def

    def __repr__(self):
        string = self.__class__.__name__
        if self.hessian_set:
            string += '\n  hessian:'
            for h in self.hessian:
                string += ('\n    ' + len(h) * '{:<8f} ').format(*tuple(h))
            #end for
            string += '\n  Conjugate directions:'
            string += '\n    Lambda     Direction'
            for Lambda, direction in zip(self.Lambda, self.get_directions()):
                string += ('\n    {:<8f}   ' + len(direction) * '{:<+1.6f} ').format(Lambda, *tuple(direction))
            #end for
        else:
            string += '\n  hessian: not set'
        #end if
        return string
    #end def

#end class


# Class for line-search along direction in abstract context
class AbstractLineSearch():

    fraction = None
    fit_kind = None
    func = None
    func_p = None
    grid = None
    values = None
    errors = None
    x0 = None
    x0_err = None
    y0 = None
    y0_err = None
    fit = None

    def __init__(
        self,
        grid = None,
        values = None,
        errors = None,
        fraction = 0.025,
        **kwargs,
    ):
        self.fraction = fraction
        self.set_func(**kwargs)
        if grid is not None:
            self.set_grid(grid)
        #end if
        if values is not None:
            self.set_values(values, errors, also_search = (self.grid is not None))
        #end if
    #end def

    def set_func(
        self,
        fit_kind = 'pf3',
        **kwargs
    ):
        self.func, self.func_p = self._get_func(fit_kind)
        self.fit_kind = fit_kind
    #end def

    def get_func(self, fit_kind = None):
        if fit_kind is None:
            return self.func, self.func_p
        else:
            return self._get_func(fit_kind)
        #end if
    #end def

    def _get_func(self, fit_kind):
        if 'pf' in fit_kind:
            func = self._pf_search
            func_p = int(fit_kind[2:])
        else:
            raise('Fit kind {} not recognized'.format(fit_kind))
        #end if
        return func, func_p
    #end def

    def set_data(self, grid, values, errors = None, also_search = True):
        self.set_grid(grid)
        self.set_values(values, errors, also_search = also_search)
    #end def

    def set_grid(self, grid):
        assert len(grid) > 2, 'Number of grid points must be greater than 2'
        self.reset()
        self.grid = array(grid)
    #end def

    def set_values(self, values, errors = None, also_search = True):
        assert len(values) == len(self.grid), 'Number of values does not match the grid'
        self.reset()
        if errors is None or all(array(errors) == None):
            self.errors = None
        else:
            self.errors = array(errors)
        #end if
        self.values = array(values)
        if also_search:
            self.search()
        #end if
    #end def

    def search(self, **kwargs):
        """Perform line-search with the preset values and settings, saving the result to self."""
        assert self.grid is not None and self.values is not None
        res = self._search_with_error(
            self.grid,
            self.values,
            self.errors,
            fit_kind = self.fit_kind,
            fraction = self.fraction,
            **kwargs)
        self.x0 = res[0]
        self.y0 = res[2]
        self.x0_err = res[1]
        self.y0_err = res[3]
        self.fit = res[4]
    #end def

    def _search(
        self,
        grid,
        values,
        fit_kind = None,
        **kwargs,
    ):
        func, func_p = self.get_func(fit_kind)
        return self._search_one(grid, values, func, func_p, **kwargs)
    #end def

    def _search_one(
        self,
        grid,
        values,
        func,
        func_p = None,
        **kwargs,
    ):
        return func(grid, values, func_p, **kwargs)  # x0, y0, fit
    #end def

    def _search_with_error(
        self,
        grid,
        values,
        errors,
        fraction = 0.25,
        fit_kind = None,
        **kwargs,
    ):
        func, func_p = self.get_func(fit_kind)
        x0, y0, fit = self._search_one(grid, values, func, func_p, **kwargs)
        # resample for errorbars
        if errors is not None:
            x0s, y0s = self._get_distribution(grid, values, errors, func = func, func_p = func_p, **kwargs)
            ave, x0_err = get_fraction_error(x0s - x0, fraction = self.fraction)
            ave, y0_err = get_fraction_error(y0s - y0, fraction = self.fraction)
        else:
            x0_err, y0_err = 0.0, 0.0
        #end if
        return x0, x0_err, y0, y0_err, fit
    #end def

    def _pf_search(
        self,
        shifts,
        values,
        pfn,
        **kwargs,
    ):
        return get_min_params(shifts, values, pfn, **kwargs)
    #end def

    def reset(self):
        self.x0, self.x0_err, self.y0, self.y0_err, self.fit = None, None, None, None, None
    #end def

    def get_x0(self, err = True):
        assert self.x0 is not None, 'x0 must be computed first'
        if err:
            return self.x0, self.x0_err
        else:
            return self.x0
        #end if
    #end def

    def get_y0(self, err = True):
        assert self.y0 is not None, 'y0 must be computed first'
        if err:
            return self.y0, self.y0_err
        else:
            return self.y0
        #end if
    #end def

    def get_distribution(self, grid = None, values = None, errors = None, **kwargs):
        grid = grid if grid is not None else self.grid
        values = values if values is not None else self.values
        errors = errors if errors is not None else self.errors
        assert errors is not None, 'Cannot produce distribution unless errors are provided'
        return self._get_distribution(grid, values, errors, **kwargs)
    #end def

    def get_x0_distribution(self, errors = None, N = 100, **kwargs):
        if errors is None:
            return array(N * [self.get_x0(err = False)])
        #end if
        return self.get_distribution(errors = errors, **kwargs)[0]
    #end def

    def get_y0_distribution(self, errors = None, N = 100, **kwargs):
        if errors is None:
            return array(N * [self.get_y0(err = False)])
        #end if
        return self.get_distribution(errors = errors, **kwargs)[1]
    #end def

    def _get_distribution(self, grid, values, errors, Gs = None, N = 100, fit_kind = None, **kwargs):
        func, func_p = self.get_func(fit_kind)
        if Gs is None:
            Gs = random.randn(N, len(errors))
        #end if
        x0s, y0s, pfs = [], [], []
        for G in Gs:
            x0, y0, pf = self._search_one(grid, values + errors * G, func, func_p)
            x0s.append(x0)
            y0s.append(y0)
            pfs.append(pf)
        #end for
        return array(x0s, dtype = float), array(y0s, dtype = float)
    #end def

    def __repr__(self):
        string = self.__class__.__name__
        if self.fit_kind is not None:
            string += '\n  fit_kind: {:s}'.format(self.fit_kind)
        #end if
        string += self.__repr_grid__()
        if self.x0 is None:
            string += '\n  x0: not set'
        else:
            x0_err = '' if self.x0_err is None else ' +/- {: <8f}'.format(self.x0_err)
            string += '\n  x0: {: <8f} {:s}'.format(self.x0, x0_err)
        #end if
        if self.y0 is None:
            string += '\n  y0: not set'
        else:
            y0_err = '' if self.y0_err is None else ' +/- {: <8f}'.format(self.y0_err)
            string += '\n  y0: {: <8f} {:s}'.format(self.y0, y0_err)
        #end if
        return string
    #end def

    # repr of grid
    def __repr_grid__(self):
        if self.grid is None:
            string = '\n  data: no grid'
        else:
            string = '\n  data:'
            values = self.values if self.values is not None else len(self.values) * ['-']
            errors = self.errors if self.errors is not None else len(self.values) * ['-']
            string += '\n    {:9s} {:9s} {:9s}'.format('grid', 'value', 'error')
            for g, v, e in zip(self.grid, values, errors):
                string += '\n    {: 8f} {:9s} {:9s}'.format(g, str(v), str(e))
            #end for
        #end if
        return string
    #end def

#end class


# Class for line-search with resampling and bias assessment against target
class AbstractTargetLineSearch(AbstractLineSearch):

    target_x0 = None
    target_y0 = None
    target_grid = None
    target_values = None
    target_xlim = None
    bias_mix = None

    def __init__(
        self,
        target_grid = None,
        target_values = None,
        target_y0 = None,
        target_x0 = 0.0,
        bias_mix = 0.0,
        **kwargs,
    ):
        AbstractLineSearch.__init__(self, **kwargs)
        self.target_x0 = target_x0
        self.target_y0 = target_y0
        self.bias_mix = bias_mix
        if target_grid is not None and target_values is not None:
            self.set_target(target_grid, target_values, **kwargs)
        #end if
    #end def

    def set_target(
        self,
        grid,
        values,
        interpolate_kind = 'cubic',
        **kwargs,
    ):
        sidx = argsort(grid)
        self.target_grid = array(grid)[sidx]
        self.target_values = array(values)[sidx]
        if self.target_y0 is None:
            self.target_y0 = self.target_values.min()  # approximation
        #end if
        self.target_xlim = [grid.min(), grid.max()]
        if interpolate_kind == 'pchip':
            self.target_in = PchipInterpolator(grid, values, extrapolate = False)
        else:
            self.target_in = interp1d(grid, values, kind = interpolate_kind, bounds_error = False)
        #end if
    #end def

    def evaluate_target(self, grid):
        assert grid.min() - self.target_xlim[0] > -1e-6 and grid.max() - self.target_xlim[1] < 1e-6, 'Requested points off the grid: ' + str(grid)
        return self.target_in(0.99999*grid)
    #end def

    def compute_bias(self, grid = None, bias_mix = None, **kwargs):
        bias_mix = bias_mix if bias_mix is not None else self.bias_mix
        grid = grid if grid is not None else self.target_grid
        return self._compute_bias(grid, bias_mix)
    #end def

    def _compute_xy_bias(self, grid, **kwargs):
        values = self.evaluate_target(grid)
        x0, y0, fit = self._search(grid, values, **kwargs)
        bias_x = x0 - self.target_x0
        bias_y = y0 - self.target_y0
        return bias_x, bias_y
    #end def

    def _compute_bias(self, grid, bias_mix = None, **kwargs):
        bias_mix = bias_mix if bias_mix is not None else self.bias_mix
        bias_x, bias_y = self._compute_xy_bias(grid, **kwargs)
        bias_tot = abs(bias_x) + bias_mix * abs(bias_y)
        return bias_x, bias_y, bias_tot
    #end def

    def compute_errorbar(
        self,
        grid = None,
        errors = None,
        **kwargs
    ):
        grid = grid if grid is not None else self.grid
        errors = errors if errors is not None else self.errors
        errorbar_x, errorbar_y = self._compute_errorbar(grid, errors, **kwargs)
    #end def

    # dvalues is an array of value fluctuations: 'errors * Gs' or 'noise * Gs'
    def _compute_errorbar(self, grid, errors, **kwargs):
        values = self.evaluate_target(grid)
        x0, x0_err, y0, y0_err, fit = self._search_with_error(grid, values, errors, **kwargs)
        return x0_err, y0_err
    #end def
    
    def compute_error(
        self,
        grid = None,
        errors = None,
        **kwargs
    ):
        bias_x, bias_y, bias_tot = self.compute_bias(grid, **kwargs)
        errorbar_x, errorbar_y = self.compute_errorbar(grid, errors, **kwargs)
        error = bias_tot + errorbar_x
        return error
    #end def

    def _compute_error(self, grid, errors, **kwargs):
        bias_x, bias_y, bias_tot = self._compute_bias(grid, **kwargs)
        errorbar_x, errorbar_y = self._compute_errorbar(grid, errors, **kwargs)
        return bias_tot + errorbar_x
    #end def

    def __repr__(self):
        string = AbstractLineSearch.__repr__(self)
        if self.target_grid is not None:
            string += '\n  target grid: set'
        #end if
        if self.target_values is not None:
            string += '\n  target values: set'
        #end if
        string += '\n  bias_mix: {:<4f}'.format(self.bias_mix)
        return string
    #end def

    # repr of grid
    # TODO: change; currently overlapping information
    def __repr_grid__(self):
        if self.target_grid is None:
            string = '\n  target data: no grid'
        else:
            string = '\n  target data:'
            values = self.target_values if self.target_values is not None else self.M * ['-']
            string += '\n    {:9s} {:9s}'.format('grid', 'value')
            for g, v in zip(self.target_grid, values):
                string += '\n    {: 8f} {:9s}'.format(g, str(v))
            #end for
        #end if
        return string
    #end def


#end class


# Class for PES line-search with recorded positions
class LineSearch(AbstractLineSearch):
    structure = None  # eqm structure
    structure_list = None  # list of LineSearchStructure objects
    d = None  # direction count
    W = None
    R = None
    M = None
    Lambda = None
    sigma = None
    # status flags
    shifted = False
    generated = False
    analyzed = False
    loaded = False
    searched = False

    def __init__(
        self,
        structure,
        hessian,
        d,
        sigma = 0.0,
        grid = None,
        **kwargs,
    ):
        self.sigma = sigma
        self.set_structure(structure)
        self.set_hessian(hessian, d)
        self.figure_out_grid(grid = grid, **kwargs)
        AbstractLineSearch.__init__(self, grid = self.grid, **kwargs)
        self.shift_structures()
    #end def

    def set_structure(self, structure):
        assert isinstance(structure, ParameterStructure), 'provided structure is not a ParameterStructure object'
        assert structure.check_consistency(), 'Provided structure is not a consistent mapping'
        self.structure = structure
    #end def

    def set_hessian(self, hessian, d):
        self.hessian = hessian
        self.Lambda = hessian.get_lambda(d)
        self.direction = hessian.get_directions(d)
        self.d = d
    #end def

    def figure_out_grid(self, **kwargs):
        self.grid, self.M = self._figure_out_grid(**kwargs)
    #end def

    def _figure_out_grid(self, M = None, W = None, R = None, grid = None, **kwargs):
        if M is None:
            M = self.M if self.M is not None else 7  # universal default
        #end if
        if grid is not None:
            self.M = len(grid)
        elif R is not None:
            assert not R < 0, 'R cannot be negative, {} requested'.format(R)
            grid = self._make_grid_R(R, M = M)
            self.R = R
        elif W is not None:
            assert not W < 0, 'W cannot be negative, {} requested'.format(W)
            grid = self._make_grid_W(W, M = M)
            self.W = W
        else:
            raise AssertionError('Must characterize grid')
        #end if
        return grid, M
    #end def

    def _make_grid_R(self, R, M):
        R = max(R, 1e-4)
        grid = linspace(-R, R, M)
        return grid
    #end def

    def _make_grid_W(self, W, M):
        R = W_to_R(max(W, 1e-4), self.Lambda)
        return self._make_grid_R(R, M = M)
    #end def

    def shift_structures(self):
        structure_list = []
        for shift in self.grid:
            structure = self._shift_structure(shift)
            structure_list.append(structure)
        #end for
        self.structure_list = structure_list
        self.shifted = True
    #end def

    def _shift_structure(self, shift, roundi = 4):
        shift_rnd = round(shift, roundi)
        params_this = self.structure.params
        if shift_rnd == 0.0:
            label = 'eqm'
            params = params_this.copy()
        else:
            sgn = '' if shift_rnd < 0 else '+'
            label = 'd{}_{}{}'.format(self.d, sgn, shift_rnd)
            params = params_this + shift * self.direction
        #end if
        structure = self.structure.copy(params = params, label = label)
        return structure
    #end def

    def generate_jobs(
        self,
        job_func,
        exclude_eqm = True,
        **kwargs,
    ):
        assert self.shifted, 'Must shift parameters first before generating jobs'
        jobs = []
        for structure in self.structure_list:
            if exclude_eqm and not structure.label == 'eqm':
                jobs += self._generate_jobs(job_func, structure, **kwargs)
            #end if
        #end for
        self.generated = True
        return jobs
    #end def

    # job must accept 0: position, 1: path, 2: noise
    def generate_eqm_jobs(
        self,
        job_func,
        sigma,
        **kwargs,
    ):
        return self._generate_jobs(job_func, self.structure, sigma = sigma, **kwargs)
    #end def

    def _make_job_path(self, path, label):
        return '{}/{}'.format(path, label)
    #end def

    # job_func must accept 0: structure, 1: path, 2: sigma
    def _generate_jobs(
        self,
        job_func,
        structure,
        sigma = None,
        path = '',
        **kwargs,
    ):
        sigma = sigma if sigma is not None else self.sigma
        path = self._make_job_path(path, structure.label)
        return job_func(structure, path = path, sigma = sigma, **kwargs)
    #end def

    # analyzer fuctions must accept 0: path
    #   return energy, errorbar
    def analyze_jobs(self, analyze_func, path = '', **kwargs):
        values, errors = [], []
        for structure in self.structure_list:
            value, error = analyze_func(self._make_job_path(path, structure.label), **kwargs)
            structure.set_value(value, error)
            values.append(value)
            errors.append(error)
        #end for
        return values, errors
    #end def

    def load_results(self, analyze_func = None, values = None, errors = None, **kwargs):
        if analyze_func is not None:
            values, errors = self.analyze_jobs(analyze_func, **kwargs)
        #end if
        self.loaded = self.set_results(values, errors, **kwargs)
        return self.loaded
    #end def

    def set_results(
        self,
        values,
        errors = None,
        **kwargs
    ):
        if values is None:
            return False
        #end if
        self.set_values(values, errors, also_search = True)
        return True
    #end def

    def get_shifted_params(self):
        return array([structure.params for structure in self.structure_list])
    #end def

    def __repr__(self):
        string = AbstractLineSearch.__repr__(self)
        string += '\n  Lambda: {:<9f}'.format(self.Lambda)
        if self.W is not None:
            string += '\n  W: {:<9f}'.format(self.W)
        #end if
        if self.R is not None:
            string += '\n  R: {:<9f}'.format(self.R)
        #end if
        return string
    #end def

    # repr of grid
    def __repr_grid__(self):
        if self.grid is None:
            string = '\n  data: no grid'
        else:
            string = '\n  data:'
            values = self.values if self.values is not None else self.M * ['-']
            errors = self.errors if self.errors is not None else self.M * ['-']
            string += '\n    {:11s}  {:9s}  {:9s}  {:9s}'.format('label', 'grid', 'value', 'error')
            for s, g, v, e in zip(self.structure_list, self.grid, values, errors):
                string += '\n    {:11s}  {: 8f}  {:9.9s}  {:<9.9s}'.format(s.label, g, str(v), str(e))
            #end for
        #end if
        return string
    #end def

#end class


# Class for error scan line-search
class TargetLineSearch(AbstractTargetLineSearch, LineSearch):

    R_max = None  # maximum R available for target evaluation
    W_max = None  # maximum W available for target evaluation
    E_mat = None  # resampled W-sigma matrix of errors
    W_mat = None  # resampled W-mesh
    S_mat = None  # resampled sigma-mesh
    T_mat = None  # resampled trust-mesh (whether error is reliable)
    Gs = None  # N x M set of correlated random fluctuations for the grid
    epsilon = None  # optimized target error
    W_opt = None  # W to meet epsilon
    sigma_opt = None  # sigma to meet epsilon
    # FLAGS
    resampled = False
    optimized = False

    def __init__(
        self,
        structure,
        hessian,
        d,
        M = 7,
        W = None,  # characteristic window
        R = None,  # max displacement
        grid = None,  # manual set of shifts
        **kwargs,
    ):
        LineSearch.__init__(
            self,
            structure,
            hessian,
            d,
            M = M,
            W = W,
            R = R,
            grid = grid,
        )
        self._set_RW_max()
        AbstractTargetLineSearch.__init__(self, **kwargs)
    #end def

    def compute_bias(self, bias_mix = None, **kwargs):
        bias_mix = bias_mix if bias_mix is not None else self.bias_mix
        grid, M = self._figure_out_grid(**kwargs)
        return self._compute_bias(grid = grid, bias_mix = bias_mix, **kwargs)
    #end def

    def compute_bias_of(
        self,
        R = None,
        W = None,
        verbose = False,
        **kwargs
    ):
        biases_x, biases_y, biases_tot = [], [], []
        if verbose:
            print((4 * '{:<10s} ').format('R', 'bias_x', 'bias_y', 'bias_tot'))
        #end if
        if R is not None:
            Rs = array([R]) if isscalar(R) else R
            for R in Rs:
                R = max(R, 1e-6)  # for numerical stability
                bias_x, bias_y, bias_tot = self.compute_bias(R = R, **kwargs)
                biases_x.append(bias_x)
                biases_y.append(bias_y)
                biases_tot.append(bias_tot)
                if verbose:
                    print((4 * '{:<10f} ').format(R, bias_x, bias_y, bias_tot))
                #end if
            #end for
        elif W is not None:
            Ws = array([W]) if isscalar(W) else W
            if verbose:
                print((4 * '{:<10s} ').format('W', 'bias_x', 'bias_y', 'bias_tot'))
            #end if
            for W in Ws:
                W = max(W, 1e-6)  # for numerical stability
                bias_x, bias_y, bias_tot = self.compute_bias(W = W, **kwargs)
                biases_x.append(bias_x)
                biases_y.append(bias_y)
                biases_tot.append(bias_tot)
                if verbose:
                    print((4 * '{:<10f} ').format(r, bias_x, bias_y, bias_tot))
                #end if
            #end for
        #end if
        return array(biases_x), array(biases_y), array(biases_tot)
    #end def

    # overrides method in LineSearch class with option to set target PES
    def set_results(self, values, errors, set_target = False, **kwargs):
        if set_target:
            self.set_target(self.grid, values, **kwargs)
            self._set_RW_max()
        else:
            LineSearch.set_results(self, values, errors, **kwargs)
        #end if
    #end def

    def evaluate_target(self, grid):
        try:
            return AbstractTargetLineSearch.evaluate_target(self, grid)
        except AssertionError:
            print('  W_max and R_max are respectively {} and {}'.format(self.W_max, self.R_max))
            return None
        #end try
    #end def

    def _set_RW_max(self):
        self.R_max = min([-self.grid.min(), self.grid.max()])
        self.W_max = R_to_W(self.R_max, self.Lambda)
    #end def

    def _W_sigma_of_epsilon(self, epsilon, **kwargs):
        W, sigma = self._contour_max_sigma(epsilon)
        return W, sigma
    #end def

    # X: grid of W values; Y: grid of sigma values; E: grid of total errors
    #   if Gs is not provided, use M and N
    def generate_W_sigma_data(
        self,
        W_num = 10,
        W_max = None,
        sigma_num = 10,
        sigma_max = None,
        Gs = None,
        M = None,
        N = None,
        fit_kind = None,
        **kwargs
    ):
        if Gs is None: 
            M = M if M is not None else self.M
            self.regenerate_Gs(M, N)  # set (or reset) Gs and M appropriately
        else: # Gs is not None
            N, M = Gs.shape
            assert N > 1, 'Must provide N > 1'
            assert M > 1, 'Must provide M > 1'
            self.Gs = Gs
            self.M = M
        #end if
        W_max = W_max if W_max is not None else self.W_max
        fit_kind = fit_kind if fit_kind is not None else self.fit_kind
        sigma_max = sigma_max if sigma_max is not None else W_max/20
        # starting window array: sigma = 0, so only bias
        Ws = linspace(0.0, W_max, W_num)
        sigmas = linspace(0.0, sigma_max, sigma_num)
        errors = array(self.M * sigmas[0])
        E_this = [self._compute_error(self._make_grid_W(W, self.M), errors, Gs = self.Gs, fit_kind = fit_kind) for W in Ws]
        self.fit_kind = fit_kind
        self.E_mat = array([E_this])
        self.W_mat = array([Ws])
        self.S_mat = array([W_num * [sigmas[0]]])
        self.T_mat = self._generate_T_mat()
        # append the rest
        for sigma in sigmas[1:]:
            self._insert_sigma_data(sigma, Gs = Gs, fit_kind = fit_kind)
        #end for
        self.resampled = True
    #end def

    def _generate_T_mat(self):
        return self.W_mat > self.S_mat
    #end def

    def _check_Gs_M_N(self, Gs, M, N):
        """Return True if Gs (and derived quantities) do not need regeneration, otherwise return False"""
        if self.Gs is None:
            return False
        #end if
        Gs = Gs if Gs is not None else self.Gs  # only Gs will matter when provided
        M = M if M is not None else self.M  # check user input
        N = N if N is not None else len(self.Gs)  # check user input
        return (len(Gs) == N and len(Gs[0]) == M)
    #end def

    def regenerate_Gs(self, M, N):
        """Regenerate and save Gs array"""
        assert N is not None and N > 1, 'Must provide N > 1'
        assert M is not None and M > 1, 'Must provide M > 1'
        self.Gs = random.randn(N, M)
        self.M = M
    #end def

    def insert_sigma_data(self, sigma, **kwargs):
        self._insert_sigma_data(sigma, Gs = self.Gs, fit_kind = self.fit_kind, **kwargs)
    #end def

    def insert_W_data(self, W, **kwargs):
        assert W < self.W_max, 'Cannot resample past W_max={:<9f}; extend the target data'.format(self.W_max)
        self._insert_W_data(W, Gs = self.Gs, fit_kind = self.fit_kind, **kwargs)
    #end def

    def _insert_sigma_data(self, sigma, **kwargs):
        Ws = self.W_mat[0]
        Es = [self._compute_error(self._make_grid_W(W, self.M), self.M * [sigma], **kwargs) for W in Ws]
        W_mat = append(self.W_mat, [Ws], axis = 0)
        S_mat = append(self.S_mat, [len(Ws) * [sigma]], axis = 0)
        E_mat = append(self.E_mat, [Es], axis = 0)
        idx = argsort(S_mat[:, 0])
        self.W_mat = W_mat[idx]
        self.S_mat = S_mat[idx]
        self.E_mat = E_mat[idx]
        self.T_mat = self._generate_T_mat()
    #end def

    def _insert_W_data(self, W, **kwargs):
        sigmas = self.S_mat[:, 0]
        grid = self._make_grid_W(W, self.M)
        Es = [self._compute_error(grid, self.M * [sigma], **kwargs) for sigma in sigmas]
        W_mat = append(self.W_mat, array([len(sigmas) * [W]]).T, axis = 1)
        S_mat = append(self.S_mat, array([sigmas]).T, axis = 1)
        E_mat = append(self.E_mat, array([Es]).T, axis = 1)
        idx = argsort(W_mat[0])
        self.W_mat = W_mat[:, idx]
        self.S_mat = S_mat[:, idx]
        self.E_mat = E_mat[:, idx]
        self.T_mat = self._generate_T_mat()
    #end def

    def optimize(self, epsilon, **kwargs):
        """Optimize W and sigma to a given target error epsilon > 0."""
        self.W_opt, self.sigma_opt = self.maximize_sigma(epsilon, **kwargs)
        self.epsilon = epsilon
        self.optimized = True
    #end def

    def maximize_sigma(
        self,
        epsilon,
        allow_override = True,  # allow regeneration of errors
        fix_res = True,
        Gs = None,
        M = None,
        N = None,
        verbose = True,
        low_thr = 0.9,
        **kwargs
    ):
        """Optimize W and sigma based on maximizing sigma."""
        if self.resampled:
            if not self._check_Gs_M_N(Gs, M, N):
                if allow_override:
                    self.generate_W_sigma_data(Gs = Gs, M = M, N = N, **kwargs)
                else:
                    raise AssertionError('Requested inconsistent resampling.')
                #end if
            #end if
        else:
            self.generate_W_sigma_data(Gs = Gs, M = M, N = N, **kwargs)
        #end if
        W, sigma, E, errs = self._maximize_y(epsilon, low_thr = low_thr)
        if 'not_found' in errs:
            raise AssertionError('W, sigma not found for epsilon = {}. Check minimum bias and raise epsilon.'.format(epsilon))
        #end if
        if fix_res:
            while 'x_underflow' in errs and self._fix_x_underflow(W, verbose = verbose):
                W, sigma, E, errs = self._maximize_y(epsilon, low_thr = low_thr)
            #end while
            while 'y_underflow' in errs and self._fix_y_underflow(sigma, verbose = verbose):
                W, sigma, E, errs = self._maximize_y(epsilon, low_thr = low_thr)
            #end while
            while 'y_overflow' in errs and self._fix_y_overflow(sigma, verbose = verbose):
                W, sigma, E, errs = self._maximize_y(epsilon, low_thr = low_thr)
            #end while
            while 'low_res' in errs and self._fix_low_res(epsilon, verbose = verbose):
                W, sigma, E, errs = self._maximize_y(epsilon, low_thr = low_thr)
            #end while
        return W, sigma
    #end def

    def _fix_x_underflow(self, W_this, verbose = True):
        W_new = self.W_mat[0, 1] / 2
        if W_new < self.W_max * 1e-3:
            if verbose:
                print('W underflow: did not add W = {}'.format(W_new.round(7)))
            #end if
            return False
        #end if
        self.insert_W_data(W_new)
        if verbose:
            print('W underflow: added W = {} to resampling grid'.format((W_new).round(7)))
        #end if
        return True
    #end def

    def _fix_y_underflow(self, S_this, verbose = True):
        S_new = self.S_mat[1, 0] / 2
        if S_new < self.W_max * 1e-8:
            return False
        #end if
        self.insert_sigma_data(S_new)
        if verbose:
            print('Sigma underflow: added sigma = {} to resampling grid'.format((S_new).round(7)))
        #end if
        return True
    #end def

    def _fix_y_overflow(self, S_this, verbose = True):
        S_new = S_this * 2
        if S_new > self.W_max:
            if verbose:
                print('sigma overflow: did not add sigma = {} to resampling grid'.format(S_new))
            #end if
            return False
        #end if
        self.insert_sigma_data(S_new)
        if verbose:
            print('Sigma overlow: added sigma = {} to resampling grid'.format((S_new).round(7)))
        #end if
        return True
    #end def

    def _fix_low_res(self, epsilon, verbose = True):
        status = False
        Wi, Si = self._argmax_y(self.E_mat, self.T_mat, epsilon)
        W_this, S_this = self.W_mat[0, Wi], self.S_mat[Si, 0]
        S_new = (self.S_mat[Si, 0] + self.S_mat[Si + 1, 0]) / 2
        W_lo = (self.W_mat[0, Wi - 1] + self.W_mat[0, Wi]) / 2
        if Wi < len(self.W_mat[0]) - 1:  # whether to add high W value
            W_hi = (self.W_mat[0, Wi] + self.W_mat[0, Wi + 1]) / 2
            if abs(W_hi - W_this) > self.W_max * 1e-3:
                status = True
                self.insert_W_data(W_hi)
                if verbose:
                    print('low-res: added W = {} to resampling grid'.format(W_hi.round(7)))
                #end if
            #end if
        #end if
        if abs(W_lo - W_this) > self.W_max * 1e-3:
            self.insert_W_data(W_lo)
            status = True
            if verbose:
                print('low-res: added W = {} to resampling grid'.format(W_lo.round(7)))
            #end if
        #end if
        if abs(S_new - S_this) > self.S_mat.max() * 1e-3:
            self.insert_sigma_data(S_new)
            status = True
            if verbose:
                print('low-res: added sigma = {} to resampling grid'.format(S_new.round(7)))
            #end if
        #end if
        return status
    #end def

    def interpolate_max_sigma(self, epsilon, low_thr = 0.9, **kwargs):
        W, sigma, E, errs = self._maximize_y(epsilon, low_thr = low_thr)
        # TODO: bilinear interpolation
        if any([err in errs for err in ['y_overflow', 'x_underflow']]):
            return W, sigma
        else:
            Wi, Si = self._argmax_y(self.E_mat, epsilon)
            a = (self.E_mat[Si + 1, Wi] - epsilon) / (self.E_mat[Si, Wi] - epsilon)
            sigma = a * (self.S_mat[Si, Wi] - self.S_mat[Si + 1, Wi])
            return W, sigma
        #end if
    #end def

    def _maximize_y(self, epsilon, low_thr = 0.9):
        """Return X, Y, and E values """
        assert self.resampled, 'Must resample errors first!'
        assert low_thr < 0.99, 'Threshold limit too high'
        assert epsilon > 0, 'epsilon must be positive'
        errs = []
        xi, yi = self._argmax_y(self.E_mat, self.T_mat, epsilon)
        if isnan(xi):
            return nan, nan, nan, ['not_found']
        #end if
        E0 = self.E_mat[yi, xi]
        x0 = self.W_mat[yi, xi]
        y0 = self.S_mat[yi, xi]
        if xi == 0:
            errs.append('x_underflow')
        elif xi == self.E_mat.shape[1] - 1:
            errs.append('x_overflow')
        #end if
        if yi == 0:
            errs.append('y_underflow')
        elif yi == self.E_mat.shape[0] - 1:
            errs.append('y_overflow')
        #end if
        if E0 / epsilon < low_thr:
            errs.append('low_res')
        #end if
        return x0, y0, E0, errs
    #end def

    def _argmax_y(self, E, T, epsilon):
        """Return indices to the highest point in E matrix that is lower than epsilon"""
        xi, yi = nan, nan
        for i in range(len(E), 0, -1):  # from high to low
            err = where((E[i - 1] < epsilon) & (T[i - 1]))
            if len(err[0]) > 0:
                yi = i - 1
                xi = err[0][argmax(E[i-1][err[0]])]
                break
            #end if
        #end for
        return xi, yi
    #end def

#end class


# Class for a bundle of parallel line-searches
class ParallelLineSearch():

    ls_list = None  # list of line-search objects
    hessian = None  # hessian object
    Lambdas = None
    directions = None
    D = None
    M = None  # number of grid points
    path = None
    windows = None
    noises = None
    fraction = None
    structure = None  # eqm structure
    structure_next = None  # next structure
    job_func = None
    analyze_func = None
    x_unit = None
    E_unit = None
    # flags
    noisy = False  # flag whether deterministic or noisy
    limits_set = False  # flag whether sensible line-search limits (windows, noises) are set
    protected  = False  # whether the structure is considered frozen and protected from changes
    generated = False  # whether structures have been generated
    loaded = False  # has loaded line-search data
    calculated = False  # new positions calculated

    def __init__(
        self,
        hessian,
        structure,
        windows = None,
        window_frac = 0.25,
        noises = None,
        path = 'pls',
        M = 7,
        fit_kind = 'pf3',
        x_unit = 'A',
        E_unit = 'Ry',
        fraction = 0.025,
        job_func = None,
        analyze_func = None,
        **kwargs,
    ):
        self.x_unit = x_unit
        self.E_unit = E_unit
        self.fraction = fraction
        self.path = path
        self.job_func = job_func
        self.analyze_func = analyze_func
        self.set_hessian(hessian)
        self.set_structure(structure)
        self.guess_windows(windows, window_frac)
        self.set_noises(noises)
        self.M = M
        self.fit_kind = fit_kind
        self.ls_list = self._generate_ls_list(**kwargs)
    #end def

    def set_hessian(self, hessian):
        self._protected()
        if isinstance(hessian, ndarray):
            hessian = ParameterHessian(hessian = hessian, x_unit = self.x_unit, E_unit = self.E_unit)
        elif isinstance(hessian, ParameterHessian):
            pass
        else:
            raise ValueError('Hessian matrix is not supported')
        #end if
        self.hessian = hessian
        self.Lambdas = hessian.Lambda
        self.directions = hessian.get_directions()
        self.D = hessian.D
    #end def

    def set_structure(self, structure):
        self._protected()
        assert isinstance(structure, ParameterStructure), 'Structure must be ParameterStructure object'
        self.structure = structure.copy(label = 'eqm')
    #end def

    def guess_windows(self, windows, window_frac):
        if windows is None:
            windows = self.Lambdas**0.5 * window_frac
            self.windows_frac = window_frac
        #end if
        self.set_windows(windows)
    #end def

    def set_windows(self, windows):
        if windows is not None:
            assert windows is not None or len(windows) == self.D, 'length of windows differs from the number of directions'
            self.windows = array(windows)
        #end if
    #end def

    def set_noises(self, noises):
        if noises is None:
            self.noisy = False
            self.noises = None
        else:
            assert(len(noises) == self.D)
            self.noisy = True
            self.noises = array(noises)
        #end if
    #end def

    def _generate_ls_list(self, **kwargs):
        noises = self.noises if self.noisy else self.D * [None]
        ls_list = []
        for d, window, noise in zip(range(self.D), self.windows, noises):
            ls = LineSearch(
                structure = self.structure,
                hessian = self.hessian,
                d = d,
                W = window,
                M = self.M,
                fit_kind = self.fit_kind,
                sigma = noise,
                **kwargs)
            ls_list.append(ls)
        #end for
        return ls_list
    #end def

    def copy(self, path, **kwargs):
        ls_args = {
            'path': path,
            'structure': self.structure,
            'hessian': self.hessian,
            'windows': self.windows,
            'noises': self.noises,
            'M': self.M,
            'fit_kind': self.fit_kind,
            'job_func': self.job_func,
            'analyze_func': self.analyze_func,
        }
        ls_args.update(**kwargs)
        ls_next = ParallelLineSearch(**ls_args)
        return ls_next
    #end def

    def propagate(self, path, protect = True, write = True, **kwargs):
        assert self.calculated, 'Must calculate data before propagating linesearch'
        pls_next = self.copy(path = path, structure = self.structure_next, **kwargs)
        self.protected = protect
        if write:
            self.write_to_disk()
        #end if
        return pls_next
    #end def

    def _protected(self):
        assert not self.protected, 'ParallelLineSearch object is protected. Change at own risk!'
    #end def

    def generate_jobs(self, job_func = None, **kwargs):
        job_func = job_func if job_func is not None else self.job_func
        sigma_min = None if not self.noisy else self.noises.min()
        eqm_jobs = self.ls_list[0].generate_eqm_jobs(job_func, path = self.path, sigma = sigma_min, **kwargs)
        jobs = eqm_jobs
        for ls in self.ls_list:
            jobs += ls.generate_jobs(job_func, path = self.path, eqm_jobs = eqm_jobs, **kwargs)
        #end for
        self.generated = True
        return jobs
    #end def

    # can either load based on analyze_func or by providing values/errors
    def load_results(self, analyze_func = None, values = None, errors = None, **kwargs):
        self._protected()
        analyze_func = analyze_func if analyze_func is not None else self.analyze_func
        values_ls = values if values is not None else self.D * [None]
        errors_ls = errors if errors is not None else self.D * [None]
        loaded = True
        for ls, values, errors in zip(self.ls_list, values_ls, errors_ls):
            loaded_this = ls.load_results(
                analyze_func = analyze_func,
                values = values,
                errors = errors,
                path = self.path,
                **kwargs)
            loaded = loaded and loaded_this
        #end for
        self.loaded = loaded
        if loaded:
            self.calculate_next()
        #end if
    #end def

    def calculate_next(self, **kwargs):
        self._protected()
        assert self.loaded, 'Must load data before calculating next parameters'
        params_next, params_next_err = self.calculate_next_params(**kwargs)
        self.structure_next = self.structure.copy(params = params_next, params_err = params_next_err)
        self.calculated = True
    #end def

    def ls(self, i):
        return self.ls_list[i]
    #end def

    def check_integrity(self):
        # TODO
        return True
    #end def

    def get_next_params(self):
        assert self.structure_next is not None, 'Next structure has not been computed yet'
        return self.structure_next.params
    #end def

    def calculate_next_params(self, **kwargs):
        directions = self.hessian.get_directions()
        structure = self.structure
        # deterministic
        params_next = self._calculate_params_next(self.get_params(), self.get_directions(), self.get_shifts())
        #stochastic
        if self.noisy:
            params_next_err = self._calculate_params_next_err(self.get_params(), self.get_directions(), params_next, **kwargs)
        else:
            params_next_err = array(self.D * [0.0])
        #end if
        return params_next, params_next_err
    #end def

    def get_params(self):
        return self.structure.params
    #end def

    def get_shifted_params(self, i = None):
        if i is None:
            return [ls.get_shifted_params() for ls in self.ls_list]
        else:
            return self.ls(i).get_shifted_params()
        #end if
    #end def

    def get_directions(self, d = None):
        return self.hessian.get_directions(d)
    #end def

    def get_shifts(self):
        assert self.loaded, 'Shift data has not been calculated yet'
        return self._get_shifts()
    #end def

    def _get_shifts(self):
        return array([ls.get_x0(err = False) for ls in self.ls_list])
    #end def

    def _calculate_params_next(self, params, directions, shifts):
        return params + shifts @ directions
    #end def

    def _calculate_params_next_err(self, params, directions, params_next, N = 200, fraction = 0.025, **kwargs):
        x0s_d = self._get_x0_distributions(N = N)
        params_d = []
        for x0s in x0s_d:
            params_d = self._calculate_params_next(params, directions, x0s) - params_next
        #end for
        params_next_err = [get_fraction_error(p, fraction = fraction) for p in array(params_d).T]
        return array(params_next_err)
    #end def

    def _get_x0_distributions(self, N = 200, **kwargs):
        return array([ls.get_x0_distribution(N = N, **kwargs) for ls in self.ls_list]).T
    #end def

    def write_to_disk(self, fname = 'data.p'):
        makedirs(self.path, exist_ok = True)
        pickle.dump(self, open(self.path + fname, mode='wb'))
    #end def

    def __repr__(self):
        string = self.__class__.__name__
        if self.ls_list is None:
            string += '\n  Line-searches: None'
        else:
            string += '\n  Line-searches:\n'
            string += indent('\n'.join(['#{:<2d} {}'.format(ls.d, str(ls)) for ls in self.ls_list]), '    ')
        #end if
        # TODO
        return string
    #end def

#end class


class TargetParallelLineSearch(ParallelLineSearch):

    epsilon_p = None
    epsilon_d = None
    error_p = None
    error_d = None
    temperature = None
    window_frac = None
    targets = None
    # FLAGS
    optimized = False

    def __init__(
        self,
        structure = None,
        hessian = None,
        targets = None,
        **kwargs
    ):
        ParallelLineSearch.__init__(self, structure = structure, hessian = hessian, **kwargs)
        self.set_targets(targets)
        self.ls_list = self._generate_tls_list(**kwargs)
    #end def

    def set_targets(self, targets):
        # whether target or not
        if targets is None:
            targets = self.D * [0.0]
        else:
            assert(len(targets) == self.D)
        #end if
        self.targets = array(targets)
    #end def

    def optimize(
        self,
        windows = None,
        noises = None,
        epsilon_p = None,
        epsilon_d = None,
        temperature = None,
        **kwargs,
    ):
        """Optimize parallel line-search for noise using different constraints. The optimizer modes are called in the following order of priority based on input parameters provided:
  1) windows, noises (list, list)
     when windows and noises per DIRECTION are provided, they are allocated to each DIRECTION and parameter errors resampled
  2) temperature (float > 0)
    temperature: windows and noises per DIRECTION are optimized to meet DIRECTIONAL tolerances based on thermal equipartition, parameter errors resampled
  3) epsilon_d (list)
    windows and noises per DIRECTION are optimized to meet DIRECTIONAL tolerances
  4) epsilon_p (list)
    windows and noises per DIRECTION are optimized for total predicted cost while meeting PARAMETER tolerances
    Guided by a keyword:
      kind = 'ls': [DEFAULT] use an auxiliary line-search method for optimization
      kind = 'thermal': find maximum temperature to maintain PARAMETER errors
      kind = 'broyden1': use broyden1 minimization to find the optimal solution (very unstable)
useful keyword arguments:
  M = number of points
  N = number of points for resampling
  fit_kind = fitting function
        """ 
        if windows is not None and noises is not None:
            self.optimize_windows_noises(windows, noises, **kwargs)
        elif temperature is not None:
            self.optimize_thermal(temperature, **kwargs)
        elif epsilon_d is not None:
            self.optimize_epsilon_d(epsilon_d, **kwargs)
        elif epsilon_p is not None:
            self.optimize_epsilon_p(epsilon_p, **kwargs)
        else:
            raise AssertionError('Not implemented')
        #end if
    #end def

    def optimize_windows_noises(self, windows, noises, M = None, fit_kind = None, **kwargs):
        M = M if M is not None else self.M
        fit_kind = fit_kind if fit_kind is not None else self.fit_kind
        self.error_d, self.error_p = self._errors_windows_noises(windows, noises, M = M, fit_kind = fit_kind, **kwargs)
        self.M = M
        self.fit_kind = fit_kind
        self.windows = windows
        self.noises = noises
        self.optimized = True
    #end def

    def _errors_windows_noises(self, windows, noises, Gs = None, fit_kind = None, M = None, **kwargs):
        Gs_d = Gs if Gs is not None else self.D * [None]
        return self._resample_errors(windows, noises, Gs = Gs_d, M = M, fit_kind = fit_kind, **kwargs)
    #end def

    def optimize_thermal(self, temperature, **kwargs):
        assert temperature > 0, 'Temperature must be positive'
        self.optimize_epsilon_d(self._get_thermal_epsilon_d(temperature), **kwargs)
    #end def

    # TODO: fixed-point method, also need to init error matrices
    def optimize_epsilon_p(
        self,
        epsilon_p,
        kind = 'ls',  # try line-search by default
        **kwargs,
    ):
        epsilon_d0 = epsilon_p.copy()  # TODO: fix
        if kind == 'ls':
            epsilon_d_opt = self._optimize_epsilon_p_ls(epsilon_p, epsilon_d0, **kwargs)
        elif kind == 'thermal':
            epsilon_d_opt = self._optimize_epsilon_p_thermal(epsilon_p, **kwargs)
        elif kind == 'broyden1':
            # Current: broyden1, probably fails to converge
            validate_epsilon_d = partial(self._resample_errors_p_of_d, target = array(epsilon_p), **kwargs)
            epsilon_d_opt = broyden1(validate_epsilon_d, epsilon_d0, f_tol = 1e-3, verbose = True)
        else:
            raise AssertionError('Fixed-point kind not recognized')
        #end if
        self.optimize_epsilon_d(epsilon_d_opt, fix_res = False)
        self.epsilon_p = epsilon_p
    #end def

    # TODO: check for the first step
    def _optimize_epsilon_p_thermal(self, epsilon_p, T0 = 0.00001, dT = 0.000005, verbose = False, **kwargs):
        T = T0
        error_p = array([-1,-1]) # init
        first = True
        while all(error_p < 0.0):
            try:
                epsilon_d = self._get_thermal_epsilon_d(T)
                error_p = self._resample_errors_p_of_d(epsilon_d, target = epsilon_p, verbose = verbose)
                if verbose:
                    print('T = {} highest error {} %'.format(T, (error_p + epsilon_p) / epsilon_p * 100))
                #end if
                T += dT
            except AssertionError:
                if verbose:
                    print('T = {} skipped'.format(T))
                #end if
                T += dT
            #end try
        #end while
        T -= 2 * dT
        return self._get_thermal_epsilon_d(T)
    #end def

    def _optimize_epsilon_p_ls(
        self,
        epsilon_p,
        epsilon_d0,
        thr = None,
        it_max = 10,
        **kwargs
    ):
        thr = thr if thr is not None else mean(epsilon_p)/20
        def cost(derror_p):
            return sum(derror_p**2)**0.5
        #end def
        epsilon_d_opt = array(epsilon_d0)
        for it in range(it_max):
            coeff = 0.5**(it + 1)
            epsilon_d_old = epsilon_d_opt.copy()
            # sequential line-search from d0...dD
            for d in range(len(epsilon_d_opt)):
                epsilon_d = epsilon_d_opt.copy()
                epsilons = linspace(epsilon_d[d] * (1 - coeff), (1 + coeff) * epsilon_d[d], 10)
                costs = []
                for s in epsilons:
                    epsilon_d[d] = s
                    derror_p = self._resample_errors_p_of_d(epsilon_d, target = epsilon_p, fix_res = False, **kwargs)
                    costs.append(cost(derror_p))
                #end for
                epsilon_d_opt[d] = epsilons[argmin(costs)]
            #end for
            derror_p = self._resample_errors_p_of_d(epsilon_d_opt, target = epsilon_p, **kwargs)
            cost_it = cost(derror_p)
            # scale down
            if cost_it < thr or sum(abs(epsilon_d_old - epsilon_d_opt)) < thr/100:
                break
            #end if
        #end for
        for c in range(100):
            if any(derror_p > 0.0):
                epsilon_d_opt = [e*0.99 for e in epsilon_d_opt]
                derror_p = self._resample_errors_p_of_d(epsilon_d_opt, target = epsilon_p, **kwargs)
            else:
                break
            #end if
        #end for
        return epsilon_d_opt
    #end def

    def optimize_epsilon_d(
        self,
        epsilon_d,
        Gs = None,
        **kwargs,
    ):
        Gs_d = Gs if Gs is not None else self.D * [None]
        assert len(Gs_d) == self.D, 'Must provide list of Gs equal to the number of directions'
        windows, noises = [], []
        for epsilon, ls, Gs in zip(epsilon_d, self.ls_list, Gs_d):
            ls.optimize(epsilon, Gs = Gs, **kwargs)
            windows.append(ls.W_opt)
            noises.append(ls.sigma_opt)
        #end for
        self.optimize_windows_noises(windows, noises, Gs = Gs_d, **kwargs)
        self.epsilon_d = epsilon_d
    #end def

    def get_Gs(self):
        return [ls.Gs for ls in self.ls_list]
    #end def

    def validate(self, N = 500, verbose = False, thr = 1.1):
        """Validate optimization by independent random resampling"""
        assert self.optimized, 'Must be optimized first'
        ref_error_p, ref_error_d = self._resample_errors(self.windows, self.noises, Gs = None, N = N)
        valid = True
        if verbose:
            print('Parameter errors:')
            print('  {:<2s}  {:<10s}  {:<10s}  {:<5s}'.format('#', 'validation', 'corr. samp.', 'ratio'))
        #end if
        for p, ref, corr in zip(range(len(ref_error_p)), ref_error_p, self.error_p):
            ratio = ref/corr
            valid_this = ratio < thr
            valid = valid and valid_this
            if verbose:
                print('  {:<2d}  {:<10f}  {:<10f}  {:<5f}'.format(p, ref, corr, ratio))
            #end if
        #end for
        if verbose:
            print('Direction errors:')
            print('  {:<2s}  {:<10s}  {:<10s}  {:<5s}'.format('#', 'validation', 'corr. samp.', 'ratio'))
        #end if
        for d, ref, corr in zip(range(len(ref_error_d)), ref_error_d, self.error_d):
            ratio = ref/corr
            valid_this = ratio < thr
            valid = valid and valid_this
            if verbose:
                print('  {:<2d}  {:<10f}  {:<10f}  {:<5f}'.format(d, ref, corr, ratio))
            #end if
        #end for
        return valid
    #end def

    def _get_thermal_epsilon_d(self, temperature):
        return [(temperature / Lambda)**0.5 for Lambda in self.Lambdas]
    #end def
    
    def _get_thermal_epsilon(self, temperature):
        return [(temperature / Lambda)**0.5 for Lambda in self.hessian.diagonal]
    #end def
    
    # Important: overrides ls_list generator
    def _generate_ls_list(self, **kwargs):
        return None
    #end def

    def _generate_tls_list(self, **kwargs):
        noises = self.noises if self.noisy else self.D * [None]
        ls_list = []
        for d, window, noise, target in zip(range(self.D), self.windows, noises, self.targets):
            tls = TargetLineSearch(
                structure = self.structure,
                hessian = self.hessian,
                d = d,
                W = window,
                sigma = noise,
                target = target,
                **kwargs)
            ls_list.append(tls)
        #end for
        return ls_list
    #end def

    def compute_bias_p(self, **kwargs):
        return self.compute_bias(**kwargs)[1]
    #end def

    def compute_bias_d(self, **kwargs):
        return self.compute_bias(**kwargs)[0]
    #end def

    def compute_bias(self, windows = None, **kwargs):
        windows = windows if windows is not None else self.windows
        return self._compute_bias(windows, **kwargs)
    #end def

    def _compute_bias(self, windows, **kwargs):
        bias_d = []
        for W, tls, target in zip(windows, self.ls_list, self.targets):
            assert W <= tls.W_max, 'window is larger than W_max'
            #bias_d.append(tls.compute_bias(W = W, **kwargs)[0] - target )
            bias_d.append(tls.compute_bias(W = W, **kwargs)[0])
        #end for
        bias_d = array(bias_d)
        bias_p = self._calculate_params_next(self.get_params(), self.get_directions(), bias_d) - self.get_params()
        return bias_d, bias_p
    #end def

    def _calculate_error(self, windows, errors, **kwargs):
        return self._calculate_params_next_error(windows, errors, **kwargs)
    #end def

    # based on windows, noises
    def _resample_errorbars(self, windows, noises, Gs = None, N = None, M = None, fit_kind = None, **kwargs):
        Gs_d = self.D * [None] if Gs is None else Gs  # provide correlated sampling
        biases_d, biases_p = self._compute_bias(windows, fit_kind = fit_kind, **kwargs)  # biases per direction, parameter
        x0s_d, x0s_p = [], []  # list of distributions of minima per direction, parameter
        errorbar_d, errorbar_p = [], []  # list of statistical errorbars per direction, parameter
        for tls, W, noise, bias_d, Gs in zip(self.ls_list, windows, noises, biases_d, Gs_d):
            assert W <= tls.W_max, 'window is larger than W_max'
            grid, M = tls._figure_out_grid(W = W, M = M)
            values = tls.evaluate_target(grid)
            errors = M * [noise]
            Gs = Gs if Gs is not None else tls.Gs
            x0s = tls.get_x0_distribution(grid = grid, values = values, errors = errors, Gs = Gs, N = N, fit_kind = fit_kind)
            x0s_d.append(x0s)
            errorbar_d.append(get_fraction_error(x0s - bias_d, self.fraction)[1])
        #end for
        # parameter errorbars
        for x0 in array(x0s_d).T:
            x0s_p.append(self._calculate_params_next(-biases_p, self.directions, x0))  # TODO: could this be vectorized?
        #end for
        errorbar_p = [get_fraction_error(x0s, self.fraction)[1] for x0s in array(x0s_p).T]
        return array(errorbar_d), array(errorbar_p)
    #end def

    def _resample_errors(self, windows, noises, **kwargs):
        bias_d, bias_p = self._compute_bias(windows, **kwargs)
        errorbar_d, errorbar_p = self._resample_errorbars(windows, noises, **kwargs)
        error_d = abs(bias_d) + errorbar_d
        error_p = abs(bias_p) + errorbar_p
        return error_d, error_p
    #end def

    def _resample_errors_p_of_d(self, epsilon_d, target = 0.0, **kwargs):
        windows, noises = self._windows_noises_of_epsilon_d(epsilon_d, **kwargs)
        return self._resample_errors(windows, noises, **kwargs)[1] - target
    #end def

    def _windows_noises_of_epsilon_d(
        self,
        epsilon_d,
        **kwargs,
    ):
        windows, noises = [], []
        for epsilon, ls, in zip(epsilon_d, self.ls_list):
            W_opt, sigma_opt = ls.maximize_sigma(epsilon, **kwargs)  # no altering the error
            #W_opt, sigma_opt = ls.interpolate_max_sigma(abs(epsilon))
            windows.append(W_opt)
            noises.append(sigma_opt)
        #end for
        return windows, noises
    #end def

#end class


# Class for line-search iteration
class LineSearchIteration():

    pls_list = []  # list of ParallelLineSearch objects
    path = ''  # base path
    n_max = None  # TODO

    def __init__(
        self,
        path = '',
        surrogate = None,
        structure = None,
        hessian = None,
        load = True,
        n_max = 0,  # no limit
        **kwargs,  # e.g. windows, noises, targets, units, job_func, analyze_func ...
    ):
        self.path = path
        self.pls_list = []
        # try to load pickles
        if load:
            self._load_until_failure()
        #end if
        if len(self.pls_list) == 0:  # if no iterations loaded, try to initialize
            # try to load from surrogate ParallelLineSearch object
            if surrogate is not None:
                self.init_from_surrogate(surrogate, **kwargs)
            #end if
            # when present, manually provided mappings, parameters and positions override those from a surrogate
            if hessian is not None and structure is not None:
                self.init_from_hessian(structure, hessian, **kwargs)
            #end if
        #end if
    #end def

    def init_from_surrogate(self, surrogate, **kwargs):
        assert isinstance(surrogate, ParallelLineSearch), 'Surrogate parameter must be a ParallelLineSearch object'
        pls = surrogate.copy(path = self._get_pls_path(0), **kwargs)
        self.pls_list = [pls]
    #end def

    def init_from_hessian(self, structure, hessian, **kwargs):
        assert isinstance(structure, ParameterStructure), 'Starting structure must be a ParameterStructure object'
        assert isinstance(hessian, ParameterHessian), 'Starting hessian must be a ParameterHessian'
        pls = ParallelLineSearch(
            path = self._get_pls_path(0),
            structure = structure,
            hessian = hessian,
            **kwargs
        )
        self.pls_list = [pls]
    #end def

    def _get_pls_path(self, i):
        return '{}pls{}/'.format(self.path, i)
    #end def

    def generate_jobs(self, **kwargs):
        return self._get_current_pls().generate_jobs(**kwargs)
    #end def

    def load_results(self, **kwargs):
        self._get_current_pls().load_results(**kwargs)
    #end def

    def _get_current_pls(self):
        return self.pls_list[-1]
    #end def

    def pls(self, i = None):
        pls = self.pls_list[i] if i is not None else self._get_current_pls()
        return pls
    #end def

    def _load_until_failure(self):
        pls_list = []
        load_failed = False
        i = 0
        while not load_failed:
            path = '{}/data.p'.format(self._get_pls_path(i))
            pls = self._load_linesearch_pickle(path)
            if pls is not None and pls.check_integrity():
                pls_list.append(pls)
                i += 1
            else:
                load_failed = True
            #end if
        #end while
        self.pls_list = pls_list
    #end def

    def _load_linesearch_pickle(self, path):
        try:
            data = pickle.load(open(path, mode='rb'))
            return data
        except FileNotFoundError:
            return None
        #end try
    #end def

    def propagate(self, **kwargs):
        pls_next = self._get_current_pls().propagate(path = self._get_pls_path(len(self.pls_list)), **kwargs)
        self.pls_list.append(pls_next)
    #end

#end class


# Dummy job function
def dummy_job(position, path, noise, **kwargs):
    pass
#end def


# Minimal function for writing line-search structures
def write_xyz_noise(position, path, noise, **kwargs):
    makedirs(path, exist_ok = True)
    savetxt('{}/structure.xyz'.format(path), position.to_xyz())
    savetxt('{}/noise'.format(path), noise)
#end def
