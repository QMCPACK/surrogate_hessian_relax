#!/usr/bin/env python3

from numpy import ndarray, array
from os import makedirs
from dill import dumps
from textwrap import indent

from lib.util import get_fraction_error
from lib.parameters import ParameterSet
from lib.hessian import ParameterHessian
from lib.linesearch import LineSearch


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
    job_args = {}
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
        job_args = {},
        analyze_func = None,
        mode = 'jobs',
        pes_func = None,
        **kwargs,
    ):
        self.x_unit = x_unit
        self.E_unit = E_unit
        self.fraction = fraction
        self.path = path
        self.job_func = job_func
        self.analyze_func = analyze_func
        self.job_args = job_args
        self.set_hessian(hessian)
        self.set_structure(structure)
        self.guess_windows(windows, window_frac)
        self.set_noises(noises)
        self.M = M
        self.fit_kind = fit_kind
        self.ls_list = self._generate_ls_list(mode = mode, pes_func = pes_func, **kwargs)
        if mode == 'pes' and pes_func is not None and 'set_target' not in kwargs:  # FIXME: refactor set_target
            self.loaded = True
            self.calculate_next(**kwargs)
        #end if
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
        assert isinstance(structure, ParameterSet), 'Structure must be ParameterSet object'
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

    def copy(self, path = '', c_noises = 1.0, **kwargs):
        if self.noises is None:
            noises = None
        else:
            noises = [noise * c_noises for noise in self.noises]
        #end if
        ls_args = {
            'path': path,
            'structure': self.structure,
            'hessian': self.hessian,
            'windows': self.windows,
            'noises': noises,
            'M': self.M,
            'fit_kind': self.fit_kind,
            'job_func': self.job_func,
            'job_args': self.job_args,
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

    def generate_jobs(self, job_func = None, job_args = {}, **kwargs):
        job_args = job_args if not job_args == {} else self.job_args
        job_func = job_func if job_func is not None else self.job_func
        sigma_min = None if not self.noisy else self.noises.min()
        eqm_jobs = self.ls_list[0].generate_eqm_jobs(job_func, path = self.path, sigma = sigma_min, **job_args, **kwargs)
        jobs = eqm_jobs
        for ls in self.ls_list:
            jobs += ls.generate_jobs(job_func, path = self.path, eqm_jobs = eqm_jobs, **job_args, **kwargs)
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
        # deterministic
        params_next = self._calculate_params_next(self.get_params(), self.get_directions(), self.get_shifts())
        # stochastic
        if self.noisy:
            params_next_err = self._calculate_params_next_error(self.get_params(), self.get_directions(), params_next, **kwargs)
        else:
            params_next_err = array(self.D * [0.0])
        #end if
        return params_next, params_next_err
    #end def

    def get_params(self):
        return self.structure.params
    #end def

    def get_params_err(self):
        err = self.structure.params_err
        if err is None:
            return self.structure.params * 0.0
        else:
            return err
        #end if
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
        return params + self._calculate_shifts(directions, shifts)
    #end def

    def _calculate_shifts(self, directions, shifts):
        return shifts @ directions
    #end def

    def _calculate_params_next_error(self, params, directions, params_next, N = 200, fraction = 0.025, **kwargs):
        x0s_d = self._get_x0_distributions(N = N)
        params_d = []
        for x0s in x0s_d:
            params_d.append(self._calculate_params_next(params, directions, x0s) - params_next)
        #end for
        params_next_err = [get_fraction_error(p, fraction = fraction)[1] for p in array(params_d).T]
        return array(params_next_err)
    #end def

    def _get_x0_distributions(self, N = 200, **kwargs):
        return array([ls.get_x0_distribution(errors = ls.errors, N = N, **kwargs) for ls in self.ls_list]).T
    #end def

    def write_to_disk(self, fname = 'data.p'):
        makedirs(self.path, exist_ok = True)
        with open(self.path + fname, mode='wb') as f:
            f.write(dumps(self))
        #end with
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
