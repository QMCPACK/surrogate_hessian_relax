#!/usr/bin/env python3

import pickle
from os import makedirs
from numpy import array, diag, linalg, linspace, savetxt
from numpy import random, argsort, isscalar, ndarray
from scipy.interpolate import interp1d, PchipInterpolator
from copy import deepcopy

from surrogate_tools import W_to_R, R_to_W, get_min_params
from surrogate_tools import get_fraction_error


Bohr = 0.5291772105638411  # A
Ry = 13.605693012183622  # eV
Hartree = 27.211386024367243  # eV


def match_to_tol(val1, val2, tol = 1e-10):
    assert len(val1) == len(val2), 'lengths of val1 and val2 do not match' + str(val1) + str(val2)
    for v1, v2 in zip(val1.flatten(), val2.flatten()):  # TODO: maybe vectorize?
        if abs(v2 - v1) > tol:
            return False
        #end if
    #end for
    return True
#end def

# TODO: currently not used
class StructuralParameter():
    kind = None
    value = None
    label = None,
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


# Base class for structural parameter mappings
class ParameterStructureBase():

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
        params = None,
        params_err = None,
        elem = None,
        periodic = False,
        dim = 3,
        value = None,
        error = None,
        label = None,
        unit = None,
        **kwargs,
    ):
        self.dim = dim
        self.periodic = periodic
        self.label = label
        if forward is not None:
            self.set_forward(forward)
        #end if
        if backward is not None:
            self.set_backward(backward)
        #end if
        if pos is not None:
            self.set_pos(pos)
        #end if
        if axes is not None:
            self.set_axes(axes)
        #end if
        if params is not None:
            self.set_params(params, params_err)
        #end if
        if elem is not None:
            self.set_elem(elem)
        #end if
        if value is not None:
            self.set_value(value, error)
        #end if
    #end def

    def set_forward(self, forward):
        self.forward_func = forward
        if self.pos is not None:
            self._set_pos(self.pos)  # rerun for forward mapping
        #end if
        self.check_consistency()
    #end def

    def set_backward(self, backward):
        self.backward_func = backward
        if self.params is not None:
            self.set_params(self.params)  # rerun for backward mapping
        #end if
        self.check_consistency()
    #end def

    def set_pos(self, pos):
        pos = array(pos)
        assert pos.size % self.dim == 0, 'Position vector inconsistent with {} dimensions!'.format(self.dim)
        self._set_pos(pos)
        if self.forward_func is not None:
            self.params = self.forward()
        #end if
        self.unset_value()
        self.check_consistency()
    #end def

    def _set_pos(self, pos):
        self.pos = array(pos).reshape(-1, self.dim)
    #end def

    def set_axes(self, axes):
        if len(axes) == self.dim:
            self.axes = diag(axes)
        else:
            axes = array(axes)
            assert axes.size == self.dim**2, 'Axes vector inconsistent with {} dimensions!'.format(self.dim)
            self.axes = array(axes).reshape(self.dim, self.dim)
        #end if
        self.unset_value()
        self.check_consistency()
    #end def

    def set_elem(self, elem):
        self.elem = elem
    #end def    

    def set_params(self, params, params_err = None):
        self.params = array(params).flatten()
        self.params_err = params_err
        if self.backward_func is not None:
            self._set_pos(self.backward(self.params))
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
        assert self.forward_func is not None, 'Forward mapping has not been supplied'
        if pos is None:
            assert self.pos is not None, 'Must supply position for forward mapping'
            pos = self.pos
        #end if
        assert not self.periodic or self.axes is not None, 'Must supply axes for periodic forward mappings'
        if self.periodic:
            return array(self.forward_func(array(pos), axes))
        else:
            return array(self.forward_func(array(pos)))
        #end if
    #end def

    def backward(self, params = None):
        assert self.backward_func is not None, 'Backward mapping has not been supplied'
        if params is None:
            assert self.params is not None, 'Must supply params for backward mapping'
            params = self.params
        #end if
        if self.periodic:
            pos, axes = self.backward_func(array(params))
            return array(pos).reshape(-1, 3), array(axes).reshape(-1, 3)
        else:
            return array(self.backward_func(array(params)))
        #end if
    #end def

    def check_consistency(
        self,
        params = None,
        pos = None,
        axes = None,
        tol = 1e-7,
    ):
        if self.forward_func is None or self.backward_func is None:
            return False
        #end if
        if pos is None and params is None:
            # if neither params nor pos are given, check internal consistency
            self.consistent = False
            if self.pos is None and self.params is None:
                # without either set of coordinates the mapping is inconsistent
                return False
            elif self.params is not None:
                if not self._check_params_consistency(self.params):
                    return False
                #end if
                if self.periodic:
                    pos, axes = self.backward()
                    if self._check_pos_consistency(pos, axes):
                        self._set_pos(pos)
                        self.consistent = True
                    #end if
                else:
                    pos = self.backward()
                    if self._check_pos_consistency(pos):
                        self._set_pos(pos)
                        self.consistent = True
                    #end if
                #end if
            else:  # self.pos is not None
                if not self._check_pos_consistency(self.pos, self.axes):
                    return False
                #end if
                params = self.forward()
                if self._check_params_consistency(params):
                    self.params = params
                    self.consistent = True
                #end if
            #end if
            return self.consistent
        elif pos is not None and params is not None:
            # if both params and pos are given, check their internal consistency
            params_new = self.forward(pos)
            pos_new = self.backward(params)
            return match_to_tol(params, params_new, tol) and match_to_tol(pos, pos_new, tol)
        elif params is not None:
            return self._check_params_consistency(params)
        else:  # pos is not None
            return self._check_pos_consistency(pos, axes)
        #end if
    #end def

    def _check_pos_consistency(self, pos, axes = None, tol = 1e-7):
        if self.periodic:
            params = self.forward(pos, axes)
            pos_new, axes_new = self.backward(params)
            consistent = match_to_tol(pos, pos_new, tol) and match_to_tol(axes, axes_new, tol)
        else:
            params = self.forward(pos)
            pos_new = self.backward(params)
            consistent = match_to_tol(pos, pos_new, tol)
        #end if
        return consistent
    #end def

    def _check_params_consistency(self, params, tol = 1e-7):
        if self.periodic:
            pos, axes = self.backward(params)
            params_new = self.forward(pos, axes)
        else:
            pos = self.backward(params)
            params_new = self.forward(pos)
        #end if
        return match_to_tol(params, params_new, tol)
    #end def

    def _shift_pos(self, dpos):
        if isscalar(dpos):
            return self.pos + dpos
        #end if
        dpos = array(dpos)
        assert self.pos.size == dpos.size
        return self.pos + dpos
    #end def

    def shift_pos(self, dpos):
        self.pos = self._shift_pos(dpos)
        if self.consistent:
            self.forward()
        #end if
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
        self.params = self._shift_params(dparams)
        if self.consistent:
            self.backward()
        #end if
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
        # TODO: assertions
        dpos = pos_ref.reshape(-1, 3) - self.pos
        return dpos
    #end def

    def jacobian(self, dp = 0.001):
        assert self.consistent, 'The mapping must be consistent'
        jacobian = []
        for p in range(len(self.params)):
            params_this = self.params.copy()
            params_this[p] += dp
            dpos = self.pos_difference(self.backward(params_this))
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
        # TODO: periodicity, cell
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
                s_args.append({
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
    U = None
    P = 0
    D = 0
    x_units = ('A', 'B')
    E_units = ('Ry', 'Ha', 'eV')
    hessian_set = False  # flag whether hessian is set (True) or just initialized (False)

    def __init__(
        self,
        hessian = None,
        hessian_real = None,
        structure = None,
        x_unit = 'A',
        E_unit = 'Ry',
    ):
        assert x_unit in self.x_units, 'x_unit {} not recognized. Available: {}'.format(x_unit, self.x_units)
        assert E_unit in self.E_units, 'E_unit {} not recognized. Available: {}'.format(E_unit, self.E_units)
        self.x_unit = x_unit
        self.E_unit = E_unit
        if structure is not None:
            if hessian_real is not None:
                self.init_hessian_real(structure, hessian_real)
            else:
                self.init_hessian_structure(structure)
            #end if
        #end if
        if hessian is not None:
            self.init_hessian_array(hessian)
        #end if
    #end def

    def init_hessian_structure(self, structure):
        assert structure.isinstance(ParameterStructure), 'Provided argument is not ParameterStructure'
        assert structure.check_consistency(), 'Provided ParameterStructure is incomplete or inconsistent'
        hessian = diag(len(structure.params) * [1.0])
        self._set_hessian(hessian)
        self.hessian_set = False  # this is not an appropriate hessian
    #end def

    def init_hessian_real(self, structure, hessian_real):
        jacobian = structure.jacobian()
        hessian = jacobian.T @ hessian_real @ jacobian
        self._set_hessian(hessian)
    #end def

    def init_hessian_array(self, hessian):
        hessian = self._convert_hessian(array(hessian))
        self._set_hessian(hessian)
    #end def

    def update_hessian_array(
        self,
        hessian,
        x_unit = 'A',
        E_unit = 'Ry',
        change_units = True,
    ):
        assert x_unit in self.x_units, 'x_unit {} not recognized. Available: {}'.format(x_unit, self.x_units)
        assert E_unit in self.E_units, 'E_unit {} not recognized. Available: {}'.format(E_unit, self.E_units)
        if change_units:
            self.x_unit = x_unit
            self.E_unit = E_unit
            #assert(x_unit == self.x_unit, 'x_unit {} does not match initial {}'.format(x_unit, self.x_unit))
            #assert(E_unit == self.E_unit, 'E_unit {} does not match initial {}'.format(E_unit, self.E_unit))
        #end if
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
        x_unit = None,
        E_unit = None,
    ):
        x_unit = self.x_unit if x_unit is None else x_unit
        E_unit = self.E_unit if E_unit is None else E_unit
        if x_unit == 'B':
            hessian *= Bohr**2
        elif x_unit == 'A':
            hessian *= 1.0
        else:
            raise ValueError('E_unit {} not recognized'.format(E_unit))
        #end if
        if E_unit == 'Ha':
            hessian /= (Ry / Hartree)**2
        elif E_unit == 'eV':
            hessian /= Ry**2
        elif E_unit == 'Ry':
            hessian *= 1.0
        else:
            raise ValueError('E_unit {} not recognized'.format(E_unit))
        #end if
        return hessian
    #end def

    def get_hessian(self, x_unit = 'A', E_unit = 'Ry'):
        return self._convert_hessian(self.hessian, x_unit = x_unit, E_unit = E_unit)
    #end def

    def __repr__(self):
        return str(self.hessian)  # TODO
    #end def

#end class


# Class for line-search along direction in abstract context
class AbstractLineSearch():

    fit_kind = None
    fraction = None
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
        self.grid = grid
    #end def

    def set_values(self, values, errors = None, also_search = True):
        assert len(values) == len(self.grid), 'Number of values does not match the grid'
        if errors is None or all(array(errors) == None):
            errors = None
        #end if
        self.values = values
        self.errors = errors
        if also_search:
            self.search()
        #end if
    #end def

    def search(self, **kwargs):
        assert self.grid is not None and self.values is not None
        res = self._search(self.grid, self.values, self.errors, fraction = self.fraction, **kwargs)
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
        errors,
        fraction = 0.025,
        fit_kind = None,
        **kwargs,
    ):
        if fit_kind is None:
            func, func_p = self.func, self.func_p
        else:
            func, func_p = self._get_func(fit_kind)
        #end if
        res0 = func(grid, values, func_p, **kwargs)
        y0 = res0[0]
        x0 = res0[1]
        fit = res0[2]
        # resample for errorbars
        if errors is not None:
            x0s, y0s = self._get_distribution(grid, values, errors, **kwargs)
            x0_err = get_fraction_error(x0s - x0, fraction = self.fraction)
            y0_err = get_fraction_error(y0s - y0, fraction = self.fraction)
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
        grid = grid if grid is None else self.grid
        values = values if values is None else self.values
        errors = errors if errors is None else self.errors
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

    def _get_distribution(self, grid, values, errors, resamples = None, N = 100, **kwargs):
        if resamples is None:
            resamples = random.randn(N, len(errors))
        #end if
        x0s, y0s, pfs = [], [], []
        for rs in resamples:
            x0, y0, pf = self.func(grid, values + errors * rs, **kwargs)
            x0s.append(x0)
            y0s.append(y0)
            pfs.append(pf)
        #end for
        return array(x0s, dtype = float), array(y0s, dtype = float)
    #end def

    def __repr__(self):
        string = self.__class__.__name__
        if self.fit_kind is None:
            string += '\n  fit_kind: {:s}'.format(self.fit_kind)
        #end if
        if self.grid is None:
            string += '\n  data: no grid'
        else:
            string += '\n  data:'
            values = self.values if self.values is not None else self.M * ['-']
            errors = self.errors if self.errors is not None else self.M * ['-']
            string += '\n    {:9s} {:9s} {:9s}'.format('grid', 'value', 'error')
            for g, v, e in zip(self.grid, values, errors):
                string += '\n    {: 8f} {:9s} {:9s}'.format(g, str(v), str(e))
            #end for
        #end if
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
        assert grid.min() > self.target_xlim[0] and grid.max() < self.target_xlim[1], 'Requested point off the grid'
        return self.target_in(grid)
    #end def

    def compute_bias(self, grid = None, bias_mix = None, **kwargs):
        bias_mix = bias_mix if bias_mix is not None else self.bias_mix
        grid = 0.9999999999*grid if grid is not None else self.target_grid
        return self._compute_bias(grid, bias_mix)
    #end def

    def _compute_xy_bias(self, grid, **kwargs):
        values = self.evaluate_target(grid)
        x0, x0_err, y0, y0_err, fit = self._search(grid, values, None, **kwargs)
        bias_x = x0 - self.target_x0
        bias_y = y0 - self.target_y0
        return bias_x, bias_y
    #end def

    def _compute_bias(self, grid, bias_mix, **kwargs):
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
        x0, x0_err, y0, y0_err, fit = self._search(grid, values, errors, **kwargs)
        return x0_err, y0_err
    #end def
    
    def compute_error(
        self,
        grid = None,
        errors = None,
        **kwargs
    ):
        bias_x, bias_y, bias_tot = self.compute_bias(grid, errors, **kwargs)
        errorbar_x, errorbar_y = self.compute_errorbar(grid, errors, **kwargs)
        error = bias_tot + errorbar_x
        return error
    #end def

    # TODO: def __repr__(self):

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

    def _figure_out_grid(self, M = 7, W = None, R = None, grid = None, **kwargs):
        if grid is not None:
            self.M = len(grid)
        elif R is not None:
            grid = self._make_grid_R(R, M = M)
            self.R = R
        elif W is not None:
            grid = self._make_grid_W(W, M = M)
            self.W = W
        else:
            raise AssertionError('Must characterize grid')
        #end if
        return grid, M
    #end def

    def _make_grid_R(self, R, M):
        assert R > 0, 'R must be positive, {} requested'.format(R)
        grid = linspace(-R, R, M)
        return grid
    #end def

    def _make_grid_W(self, W, M):
        assert W > 0, 'W must be positive, {} requested'.format(W)
        R = W_to_R(W, self.Lambda)
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

#end class


# Class for error scan line-search
class TargetLineSearch(LineSearch, AbstractTargetLineSearch):

    R_max = None
    W_max = None

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
        AbstractTargetLineSearch.__init__(self, **kwargs)
    #end def

    def compute_bias(self, bias_mix = None, **kwargs):
        bias_mix = bias_mix if bias_mix is not None else self.bias_mix
        grid, M = self._figure_out_grid(**kwargs)
        return self._compute_bias(grid = grid, bias_mix = bias_mix, **kwargs)
    #end def

    def compute_errorbar(self, sigma = None, errors = None, **kwargs):
        sigma = sigma if not sigma is None else self.sigma
        grid, M = self._figure_out_grid(**kwargs)
        errors = errors if not errors is None else array(M * [sigma])
        return self._compute_errorbar(grid = grid, errors = errors, **kwargs)
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

#end class


# Class for a bundle of parallel line-searches
class ParallelLineSearch():

    ls_list = None  # list of line-search objects
    hessian = None  # hessian object
    Lambdas = None
    directions = None
    D = None
    path = None
    windows = None
    noises = None
    surrogate = None  # surrogate line-search object
    fraction = None
    structure = None  # eqm structure
    structure_next = None  # next structure
    job_func = None
    analyze_func = None
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
        x_units = 'A',
        E_units = 'Ry',
        fraction = 0.025,
        job_func = None,
        analyze_func = None,
        **kwargs,
    ):
        self.x_units = x_units
        self.E_units = E_units
        self.fraction = fraction
        self.path = path
        self.job_func = job_func
        self.analyze_func = analyze_func
        self.set_hessian(hessian)
        self.set_structure(structure)
        self.guess_windows(windows, window_frac)
        self.set_noises(noises)
        self._generate_ls_list(**kwargs)
    #end def

    def set_hessian(self, hessian):
        self._protected()
        if isinstance(hessian, ndarray):
            hessian = ParameterHessian(hessian = hessian, x_units = self.x_units, E_units = self.E_units)
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
                sigma = noise,
                **kwargs)
            ls_list.append(ls)
        #end for
        self.ls_list = ls_list
    #end def

    def copy(self, path, **kwargs):
        ls_args = {
            'path': path,
            'structure': self.structure,
            'hessian': self.hessian,
            'windows': self.windows,
            'noises': self.noises,
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
            params_next_err = self.calculate_params_next_err(self.get_params(), self.get_directions(), params_next, **kwargs)
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

#end class


class TargetParallelLineSearch(ParallelLineSearch):

    epsilon = None
    epsilon_d = None
    temperature = None
    window_frac = None

    def __init__(
        self,
        structure = None,
        hessian = None,
        targets = None,
        **kwargs
    ):
        ParallelLineSearch.__init__(self, structure = structure, hessian = hessian, **kwargs)
        self.set_targets(targets)
        self.set_tolerances(**kwargs)
        self._generate_tls_list(**kwargs)
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

    def set_results(self, grid = None, values = None, set_targets = True, **kwargs):
        grid = grid if grid is not None else self.grid
    #end def

    def set_tolerances(
        self,
        epsilon = None,
        epsilon_d = None,
        **kwargs
    ):
        # TODO: assertions
        self.epsilon = epsilon
        self.epsilon_d = epsilon_d
    #end def

    def optimize(
        self,
        epsilon = None,
        epsilon_d = None,
        temperature = None,
        mixer_func = None,
        **kwargs,
    ):
        if temperature is not None:
            self.optimize_thermal(temperature, **kwargs)
        else:
            raise AssertionError('Not implemented')
        #end if
    #end def

    def optimize_thermal(
        self,
        temperature,
        **kwargs,
    ):
        assert temperature > 0, 'Temperature must be positive'
        epsilon_d = self._get_thermal_epsilond(temperature)
    #end def

    def _get_thermal_epsilon_d(self, temperature):
        return [(temperature / Lambda)**0.5 for Lambda in self.Lambdas]
    #end def
    
    def _get_thermal_epsilon(self, temperature):
        return [(temperature / Lambda)**0.5 for Lambda in self.hessian.diagonal]
    #end def
    
    # Important: overrides ls_list generator
    def _generate_ls_list(self, **kwargs):
        pass
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
        self.ls_list = ls_list
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
            bias_d.append(tls.compute_bias(W = W, **kwargs)[0] - target )
        #end for
        bias_d = array(bias_d)
        bias_p = self._calculate_params_next(self.get_params(), self.get_directions(), bias_d) - self.get_params()
        return bias_d, bias_p
    #end def

    def _calculate_error(self, windows, errors, **kwargs):
        return self._calculate_params_next_error(windows, errors, **kwargs) - self.targets
    #end def

    # based on windows, noises
    def _resample_errorbars(self, windows, noises, N = 200, **kwargs):
        x0s_d = []
        biases_d, biases_p = self._compute_bias(windows, **kwargs)
        errorbar_d, errorbar_p = [], []
        for W, noise, tls, bias_d in zip(windows, noises, tls, biases_d):
            grid, M = self._figure_out_grid(W = W)
            values = self.evaluate_target(grid)
            errors = M * [noise]
            x0s = ls._get_x0_distribution(grid = grid, value = values, errors = errors, N = N)
            x0s_d.append(x0s)
            errorbar_d.append(get_fraction_error(x0s - bias_d), **kwargs)
        #end for
        # parameter errorbars
        for x0s in array(x0s_d).T:
            params_d = self._calculate_params_next(-bias_p, directions, x0s)
        #end for
        errorbar_p = [get_fraction_error(p - biases_p, **kwargs) for p in array(params_d).T]
        return array(errorbar_d), array(errorbar_p)
    #end def

    def _resample_errors(self, windows, noises, **kwargs):
        bias_d, bias_p = self._compute_bias(windows, **kwargs)
        errorbar_d, errorbar_p = self._resample_errorbars(windows, noises, **kwargs)
        error_d = abs(bias_d) + errorbar_d
        error_p = abs(bias_p) + errorbar_p
        return error_d, error_p
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
        no_load = False,
        n_max = 0,  # no limit
        **kwargs,  # e.g. windows, noises, targets, units, job_func, analyze_func ...
    ):
        self.path = path
        self.pls_list = []
        # try to load pickles
        if not no_load:
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
