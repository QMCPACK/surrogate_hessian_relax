#!/usr/bin/env python3

from numpy import array, linspace, random, concatenate, polyval
from matplotlib import pyplot as plt

from lib.parameters import ParameterSet
from lib.util import get_min_params, get_fraction_error, W_to_R


# Class for line-search along direction in abstract context
class LineSearchBase():

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
            self.set_values(grid, values, errors, also_search = (self.grid is not None))
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

    def set_grid(self, grid):
        assert len(grid) > 2, 'Number of grid points must be greater than 2'
        self.reset()
        self.grid = array(grid)
    #end def

    def set_values(self, grid = None, values = None, errors = None, also_search = True):
        grid = grid if grid is not None else self.grid
        assert values is not None, 'must set values'
        assert len(values) == len(grid), 'Number of values does not match the grid'
        self.reset()
        if errors is None:
            self.errors = None
        else:
            self.errors = array(errors)
        #end if
        self.set_grid(grid)
        self.values = array(values)
        if also_search and not all(array(values) == None):
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
        self.analyzed = True
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
        grid,
        values,
        pfn,
        **kwargs,
    ):
        return get_min_params(grid, values, pfn, **kwargs)
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

    # TODO: refactor to support generic fitting functions
    def val_data(self, xdata):
        if self.fit is None:
            return None
        else:
            return polyval(self.fit, xdata)
        #end def
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

    def __str__(self):
        string = self.__class__.__name__
        if self.fit_kind is not None:
            string += '\n  fit_kind: {:s}'.format(self.fit_kind)
        #end if
        string += self.__str_grid__()
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

    # str of grid
    def __str_grid__(self):
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


# Class for PES line-search with recorded positions
class LineSearch(LineSearchBase):
    structure = None  # eqm structure
    structure_list = None  # list of LineSearchStructure objects
    jobs_list = None  # boolean list for bookkeeping of jobs
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

    def __init__(
        self,
        structure,
        hessian,
        d,
        sigma = 0.0,
        grid = None,
        **kwargs,
    ):
        self.sigma = sigma if sigma is not None else 0.0
        self.set_structure(structure)
        self.set_hessian(hessian, d)
        self.figure_out_grid(grid = grid, **kwargs)
        LineSearchBase.__init__(self, grid = self.grid, **kwargs)
        self.shift_structures()  # shift
    #end def

    def set_structure(self, structure):
        assert isinstance(structure, ParameterSet), 'provided structure is not a ParameterSet object'
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
        jobs_list = []
        for shift in self.grid:
            structure = self._shift_structure(shift)
            structure_list.append(structure)
            jobs_list.append(False)
        #end for
        self.structure_list = structure_list
        self.jobs_list = jobs_list
        self.shifted = True
    #end def

    def add_shift(self, shift):
        structure = self._shift_structure(shift)
        self.structure_list.append(structure)
        self.jobs_list.append(False)
        self.grid = concatenate([self.grid, [shift]])
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

    def evaluate_pes(
        self,
        pes_func,
        pes_args = {},
        **kwargs,
    ):
        grid, values, errors = [], [], []
        for shift, structure in zip(self.grid, self.structure_list):
            value, error = pes_func(structure, sigma = self.sigma, **pes_args)
            grid.append(shift)
            values.append(value)
            errors.append(error)
        #end for
        return array(grid), array(values), array(errors)
    #end def

    def generate_jobs(
        self,
        pes_func,
        exclude_eqm = True,
        **kwargs,
    ):
        assert self.shifted, 'Must shift parameters first before generating jobs'
        jobs = []
        for si, structure in enumerate(self.structure_list):
            if self.jobs_list[si]:
                continue
            else:
                self.jobs_list[si] = True
            #end if
            if exclude_eqm and not structure.label == 'eqm':
                s = structure.copy()
                s.to_nexus_only()
                jobs += self._generate_jobs(pes_func, s, **kwargs)
            #end if
        #end for
        self.generated = True
        return jobs
    #end def

    # job must accept 0: position, 1: path, 2: sigma
    def generate_eqm_jobs(
        self,
        pes_func,
        sigma,
        **kwargs,
    ):
        if self.generated:
            return []
        #end if
        structure = self.structure.copy()  # copy to be safe
        structure.to_nexus_only()
        return self._generate_jobs(pes_func, structure, sigma = sigma, **kwargs)
    #end def

    def _make_job_path(self, path, label):
        return '{}{}'.format(path, label)
    #end def

    # pes_func must accept 0: structure, 1: path, 2: sigma
    def _generate_jobs(
        self,
        pes_func,
        structure,
        sigma = None,
        path = '',
        **kwargs,
    ):
        sigma = sigma if sigma is not None else self.sigma
        path = self._make_job_path(path, structure.label)
        return pes_func(structure, path = path, sigma = sigma, **kwargs)
    #end def

    def analyze_job(self, label, load_func, load_args = {}, path = '', add_sigma = False, sigma = None, **kwargs):
        value, error = load_func(self._make_job_path(path, label), **load_args)
        sigma = sigma if sigma is not None else self.sigma
        if add_sigma:
            error += sigma
            value += sigma * random.randn(1)[0]
        #end if
        return value, error
    #end def

    # analyzer fuctions must accept 0: path
    #   return energy, errorbar
    def analyze_jobs(self, load_func, prune0 = True, **kwargs):
        grid, values, errors = [], [], []
        for shift, structure in zip(self.grid, self.structure_list):
            value, error = self.analyze_job(structure.label, load_func = load_func, **kwargs)
            structure.set_value(value, error)
            # FIXME: skipping values messes up the grid <-> list consistency
            if prune0 and value == 0:
                print('ls{}: skipped shift = {}, value {}'.format(self.d, shift, value))
            else:
                grid.append(shift)
                values.append(value)
                errors.append(error)
            #end if
        #end for
        return array(grid), array(values), array(errors)
    #end def

    def load_results(self, load_func = None, grid = None, values = None, errors = None, **kwargs):
        if load_func is not None:
            grid, values, errors = self.analyze_jobs(load_func, **kwargs)
        else:
            # allow to input only values, not grid
            grid = grid if grid is not None else self.grid
        #end if
        self.loaded = self.set_results(grid, values, errors, **kwargs)
        return self.loaded
    #end def

    def load_eqm_results(self, load_func = None, values = None, errors = None, **kwargs):
        if load_func is not None:
            value, error = self.analyze_job('eqm', load_func, **kwargs)
        else:
            value, error = values, errors
        #end if
        return value, error
    #end def

    def set_results(
        self,
        grid,
        values,
        errors = None,
        **kwargs
    ):
        if values is None or all(array(values) == None):
            return False
        #end if
        if errors is None:
            errors = 0.0 * array(values)
        #end if
        self.set_values(grid, values, errors, also_search = True)
        self._update_list_values(values, errors)
        return True
    #end def

    def _update_list_values(self, values, errors):
        for s, v, e in zip(self.structure_list, values, errors):
            s.value = v
            s.value_err = e
        #end for
    #end def

    def get_shifted_params(self):
        return array([structure.params for structure in self.structure_list])
    #end def

    def plot(
        self,
        ax = None,
        figsize = (4, 3),
        color = 'tab:blue',
        linestyle = '-',
        marker = '.',
        return_ax = False,
        c_lambda = 1.0,  # FIXME: replace with unit conversions
        **kwargs
    ):
        if ax is None:
            f, ax = plt.subplots()
        #end if
        xdata = self.grid
        xmin = xdata.min()
        xmax = xdata.max()
        xlen = xmax - xmin
        xlims = [xmin - xlen / 8, xmax + xlen / 8]
        xllims = [xmin + xlen / 8, xmax - xlen / 8]
        xgrid = linspace(xlims[0], xlims[1], 201)
        xlgrid = linspace(xllims[0], xllims[1], 201)
        ydata = self.values
        edata = self.errors
        x0 = self.x0
        y0 = self.y0
        pfl = [self.Lambda / 2 * c_lambda, -x0, y0]
        # plot lambda
        stylel_args = {'color': color, 'linestyle': ':'}  # etc
        ax.plot(xlgrid, polyval(pfl, xlgrid), **stylel_args)
        # plot the line-search data
        style1_args = {'color': color, 'linestyle': 'None', 'marker': marker}  # etc
        style2_args = {'color': color, 'linestyle': linestyle, 'marker': 'None'}
        if edata is None or all(array(edata) == None):
            ax.plot(xdata, ydata, **style1_args)
        else:
            ax.errorbar(xdata, ydata, edata, **style1_args)
        #end if
        ax.plot(xgrid, self.val_data(xgrid), **style2_args)
        if return_ax:
            return ax
        #end if
    #end def

    def __str__(self):
        string = print(LineSearchBase)
        string += '\n  Lambda: {:<9f}'.format(self.Lambda)
        if self.W is not None:
            string += '\n  W: {:<9f}'.format(self.W)
        #end if
        if self.R is not None:
            string += '\n  R: {:<9f}'.format(self.R)
        #end if
        return string
    #end def

    # str of grid
    def __str_grid__(self):
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
