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
#end class


# Class for structural parameter mappings
class ParameterMapping():

    forward_func = None  # mapping function from pos to params
    backward_func = None  # mapping function from params to pos
    pos = None  # real-space position
    axes = None  # cell axes
    params = None  # reduced parameters
    params_err = None  # store parameter uncertainties
    dim = None  # dimensionality
    elem = None  # list of elements
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
        label = '',
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
            self.set_pos(self.pos)  # rerun for forward mapping
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
        self.pos = pos.reshape(-1, self.dim)
        if self.forward_func is not None:
            self.params = self.forward(self.pos)
        #end if
        self.unset_value()
        self.check_consistency()
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

    def set_params(self, params, params_err = None):
        self.params = array(params).flatten()
        self.params_err = params_err
        if self.backward_func is not None:
            self.pos = self.backward(self.params)
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
            return array(pos), array(axes)
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
                        self.pos = pos
                        self.consistent = True
                    #end if
                else:
                    pos = self.backward()
                    if self._check_pos_consistency(pos):
                        self.pos = pos
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
        **kwargs,
    ):
        structure = deepcopy(self)
        if params is not None:
            structure.unset_value()
            structure.set_params(params)
        #end if
        if label is not None:
            structure.label = label
        #end if
        return structure
    #end def

#end class


# Class for physical structure (Nexus)
try:
    from structure import Structure

    class LineSearchStructure(ParameterMapping, Structure):
        kind = 'nexus'
    #end class
except ModuleNotFoundError:  # plain implementation if nexus not present
    class LineSearchStructure(ParameterMapping):
        kind = 'regular'
    #end class
#end try


# Class for Hessian matrix
class LineSearchHessian():
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
        structure = None,
        x_unit = 'A',
        E_unit = 'Ry',
    ):
        assert x_unit in self.x_units, 'x_unit {} not recognized. Available: {}'.format(x_unit, self.x_units)
        assert E_unit in self.E_units, 'E_unit {} not recognized. Available: {}'.format(E_unit, self.E_units)
        self.x_unit = x_unit
        self.E_unit = E_unit
        if structure is not None:
            self.init_hessian_structure(structure)
        #end if
        if hessian is not None:
            self.init_hessian_array(hessian)
        #end if
    #end def

    def init_hessian_structure(self, structure):
        assert structure.isinstance(LineSearchStructure), 'Provided argument is not LineSearchStructure'
        assert structure.check_consistency(), 'Provided LineSearchStructure is incomplete or inconsistent'
        hessian = diag(len(structure.params) * [1.0])
        Lambda, U = linalg.eig(hessian)
        self.P, self.D = self.hessian.shape
        self.Lambda, self.U = Lambda, U
        self.hessian_set = False
    #end def

    def init_hessian_array(self, hessian):
        self.hessian = self._convert_hessian(array(hessian))
        if len(self.hessian) == 1:
            Lambda = array(self.hessian[0])
            U = array([1.0])
            self.P, self.D = 1, 1
        else:
            Lambda, U = linalg.eig(self.hessian)
            self.P, self.D = self.hessian.shape
        #end if
        # TODO: various tests
        self.Lambda, self.U = Lambda, U
        self.hessian_set = True
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
        # TODO: tests
        self.hessian = hessian
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

#end class


# Class for line-search along direction in abstract context
class AbstractLineSearch():

    fit_kind = None
    pfn = None
    errorbar = None
    func = None
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
        fit_kind = 'pf3',
        pfn = 3,
        errorbar = 0.025,
        **kwargs,
    ):
        if 'pf' in fit_kind:
            self.func = self._pf_search
            try:
                self.pfn = int(fit_kind[2:])
            except ValueError:
                self.pfn = pfn
            #end try
        else:
            raise('Fit kind {} not recognized'.format(fit_kind))
        #end if
        self.fit_kind = fit_kind
        self.errorbar = errorbar
        if grid is not None:
            self.set_grid(grid)
        #end if
        if values is not None:
            self.set_values(values, errors, also_search = (self.grid is not None), **kwargs)
        #end if
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
        res = self._search(self.grid, self.values, self.errors, **kwargs)
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
        **kwargs
    ):
        res0 = self.func(grid, values, **kwargs)
        y0 = res0[0]
        x0 = res0[1]
        fit = res0[2]
        # resample for errorbars
        if errors is not None:
            x0s, y0s = self.get_distribution(grid, values, errors, **kwargs)
            x0_err = get_fraction_error(x0s, fraction = self.errorbar)
            y0_err = get_fraction_error(y0s, fraction = self.errorbar)
        else:
            x0_err, y0_err = 0.0, 0.0
        #end if
        return x0, x0_err, y0, y0_err, fit
    #end def

    def _pf_search(
        self,
        shifts,
        energies,
        pfn = None,
        **kwargs,
    ):
        if pfn is None:
            pfn = self.pfn
        #end if
        return get_min_params(shifts, energies, pfn, **kwargs)
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
        interpolate_kind = 'pchip',
        **kwargs,
    ):
        AbstractLineSearch.__init__(self, **kwargs)
        self.target_x0 = target_x0
        self.target_y0 = target_y0
        self.bias_mix = bias_mix
        if target_grid is not None and target_values is not None:
            self.set_target(target_grid, target_values, interpolate_kind = interpolate_kind)
        #end if
    #end def

    def set_target(
        self,
        grid,
        values,
        interpolate_kind = 'pchip',
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

    def evaluate_bias(self, grid, values, **kwargs):
        x0, x0_err, y0, y0_err, fit = self._search(grid, values, None)
        bias_x = x0 - self.target_x0
        bias_y = y0 - self.target_y0
        bias_tot = abs(bias_x) + self.bias_mix * abs(bias_y)
        return bias_x, bias_y, bias_tot
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
        M = 7,
        W = None,  # characteristic window
        R = None,  # max displacement
        shifts = None,  # manual set of shifts
        sigma = 0.0,
        **kwargs,
    ):
        self.sigma = sigma
        self.set_structure(structure)
        self.set_hessian(hessian, d)
        self.figure_out_grid(M = M, W = W, R = R, shifts = shifts)
        AbstractLineSearch.__init__(self, grid = self.grid, **kwargs)
        self.shift_structures()
    #end def

    def set_structure(self, structure):
        assert isinstance(structure, LineSearchStructure), 'provided structure is not a LineSearchStructure object'
        assert structure.check_consistency(), 'Provided structure is not a consistent mapping'
        self.structure = structure
    #end def

    def set_hessian(self, hessian, d):
        self.hessian = hessian
        self.Lambda = hessian.get_lambda(d)
        self.direction = hessian.get_directions(d)
        self.d = d
    #end def

    def figure_out_grid(self, M = 7, W = None, R = None, shifts = None):
        if shifts is not None:
            grid = shifts
            self.M = len(shifts)
        elif R is not None:
            grid = self._make_grid_R(R, M = M)
            self.R = R
        elif W is not None:
            grid = self._make_grid_W(W, M = M)
            self.W = W
        else:
            raise AssertionError('Must characterize grid')
        #end if
        self.grid = grid
        self.M = M
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
        return '{}{}'.format(path, label)
    #end def

    # job_func must accept 0: position, 1: path, 2: sigma
    def _generate_jobs(
        self,
        job_func,
        structure,
        sigma = None,
        path = '',
        **kwargs,
    ):
        sigma = sigma if sigma is not None else self.sigma
        pos = structure.pos
        path = self._make_job_path(path, structure.label)
        return job_func(pos, path, sigma, **kwargs)
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

#end class


# Class for error scan line-search
class TargetLineSearch(LineSearch, AbstractTargetLineSearch):

    def __init__(
        self,
        structure,
        hessian,
        d,
        M = 7,
        W = None,  # characteristic window
        R = None,  # max displacement
        shifts = None,  # manual set of shifts
        **kwargs,
    ):
        AbstractTargetLineSearch.__init__(self, **kwargs)
        LineSearch.__init__(
            self,
            structure,
            hessian,
            d,
            M = M,
            W = W,
            R = R,
            shifts = shifts,
        )
    #end def

    def compute_bias_R(
        self,
        Rs,
        verbose = False,
        M = 7,
        **kwargs
    ):
        if isscalar(Rs):
            Rs = array([Rs])
        #end if
        biases_x, biases_y, biases_tot = [], [], []
        if verbose:
            print((4 * '{:<10s} ').format('R', 'bias_x', 'bias_y', 'bias_tot'))
        #end if
        for R in Rs:
            grid = self._make_grid_R(R, M = M)
            values = self.evaluate_target(grid)
            bias_x, bias_y, bias_tot = self.evaluate_bias(grid, values)
            biases_x.append(bias_x)
            biases_y.append(bias_y)
            biases_tot.append(bias_tot)
            if verbose:
                print((4 * '{:<10f} ').format(R, bias_x, bias_y, bias_tot))
            #end if
        #end for
        return array(biases_x), array(biases_y), array(biases_tot)
    #end def

    def compute_bias_W(self, W, **kwargs):
        R = W_to_R(W, self.Lambda)
        return self.compute_bias_R(R, **kwargs)
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
    targets = False  # whether resampling versus target or not
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
        noises = None,
        targets = None,
        path = 'pls',
        x_units = 'A',
        E_units = 'Ry',
        Lambda_frac = 0.2,  # max window fraction of Lambda
        noise_frac = 0.0,  # max noise fraction of window
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
        self.Lambda_frac = Lambda_frac
        self.noise_frac = noise_frac
        self.set_windows(windows)
        self.set_noises(noises)
        self.set_targets(targets)
        self._generate(**kwargs)
    #end def

    def set_hessian(self, hessian):
        self._protected()
        if isinstance(hessian, ndarray):
            hessian = LineSearchHessian(hessian = hessian, x_units = self.x_units, E_units = self.E_units)
        elif isinstance(hessian, LineSearchHessian):
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
        assert isinstance(structure, LineSearchStructure), 'Structure must be LineSearchStructure object'
        self.structure = structure.copy(label = 'eqm')
    #end def

    def set_windows(self, windows):
        # guess windows
        if windows is None:
            windows = self.Lambdas * self.Lambda_frac
        else:
            assert(len(windows) == self.D), 'length of windows differs from the number of directions'
        #end if
        self.windows = array(windows)
    #end def

    def set_noises(self, noises):
        if noises is None or all(array(noises) is None):
            if self.noise_frac > 0.0:
                self.noises = self.Lambdas * self.Lambda_frac * self.noise_frac
                self.noisy = True
            else:
                self.noisy = False
                self.noises = self.D * [None]
            #end if
        else:
            assert(len(noises) == self.D)
            self.noises = array(noises)
            self.noisy = True
        #end if
    #end def

    def set_targets(self, targets):
        # whether target or not
        if targets is None:
            targets = self.D * [False]
        else:
            assert(len(targets) == self.D)
        #end if
        self.targets = targets
    #end def

    def _generate(self, **kwargs):
        ls_list = []
        for Lambda, direction, d, window, noise, target in zip(self.Lambdas, self.directions, range(self.D), self.windows, self.noises, self.targets):
            if target:
                ls = TargetLineSearch(
                    structure = self.structure,
                    hessian = self.hessian,
                    d = d,
                    W = window,
                    target = target,
                    sigma = noise,
                    **kwargs)
            else:
                ls = LineSearch(
                    structure = self.structure,
                    hessian = self.hessian,
                    d = d,
                    W = window,
                    sigma = noise,
                    **kwargs)
            #end if
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
        params_next, params_next_err = self._calculate_params_with_errs(**kwargs)
        self.structure_next = self.structure.copy(params = params_next, params_err = params_next_err)
        self.calculated = True
    #end def

    def check_integrity(self):
        # TODO
        return True
    #end def

    def get_next_params(self):
        assert self.structure_next is not None, 'Next structure has not been computed yet'
        return self.structure_next.params
    #end def

    def _calculate_params_with_errs(self, N = 100, **kwargs):
        directions = self.hessian.get_directions()
        structure = self.structure
        # deterministic
        x0s = []
        for ls in self.ls_list:
            x0s.append(ls.get_x0(err = False))
        #end for
        params = self._calculate_param_next(structure.params, directions, x0s)
        #stochastic
        x0s_d = []
        resample = False
        for ls in self.ls_list:
            x0s_d.append(ls.get_x0_distribution(N = N))
            if ls.errors is not None:
                resample = True
            #end if
        #end for
        if resample:
            params_d = []
            for x0s in array(x0s_d).transpose():
                params_d.append(self._calculate_param_next(structure.params, directions, x0s))
            #end for
            params_err = []
            for param_d in params_d:
                params_err.append(get_fraction_error(param_d, fraction = self.fraction))
            #end
        else:
            params_err = self.D * [None]
        #end if
        return params, params_err
    #end def

    def _calculate_param_next(self, params, directions, shifts):
        return params + shifts @ directions
    #end def

    def write_to_disk(self, fname = 'data.p'):
        makedirs(self.path, exist_ok = True)
        pickle.dump(self, open(self.path + fname, mode='wb'))
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
        assert isinstance(structure, LineSearchStructure), 'Starting structure must be a LineSearchStructure object'
        assert isinstance(hessian, LineSearchHessian), 'Starting hessian must be a LineSearchHessian'
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
