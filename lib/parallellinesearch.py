#!/usr/bin/env python3

from numpy import ndarray, array
from os import makedirs, path
from dill import dumps
from textwrap import indent

from lib.util import get_fraction_error, directorize
from lib.parameters import ParameterSet
from lib.hessian import ParameterHessian
from lib.linesearch import LineSearch
from lib.pessampler import PesSampler


# Class for a bundle of parallel line-searches
class ParallelLineSearch(PesSampler):

    ls_list = []  # list of line-search objects
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
    x_unit = None
    E_unit = None
    noisy = False  # flag whether deterministic or noisy
    msg_setup = 'Setup: first set_structure() and set_hessian() with valid input'
    msg_shiftep = 'Shifted: first set_windows() to define displacements'
    msg_loaded = 'Loaded: first load_results() with valid input'

    def __init__(
        self,
        hessian = None,
        structure = None,
        windows = None,
        window_frac = 0.25,
        noises = None,
        path = 'pls',
        M = 7,
        fit_kind = 'pf3',
        x_unit = 'A',
        E_unit = 'Ry',
        fraction = 0.025,
        mode = 'pes',  # (nexus|files|pes)
        shift_params = None,
        **kwargs,
    ):
        PesSampler.__init__(self, mode, **kwargs)
        self.x_unit = x_unit
        self.E_unit = E_unit
        self.path = directorize(path)
        self.fraction = fraction
        self.M = M
        self.fit_kind = fit_kind
        if structure is not None:
            self.set_structure(structure, shift_params)
        #end if
        if hessian is not None:
            self.set_hessian(hessian)
        #end if
        if self.status.setup:
            self.guess_windows(windows, window_frac)
            self.set_noises(noises)
        #end if
    #end def

    def _setup(self):
        # test for proper hessian and structure
        if isinstance(self.hessian, ParameterHessian) and isinstance(self.structure, ParameterSet):
            # test for compatibility
            if not len(self.hessian.Lambda) == len(self.structure.params):
                return False
            #end if
        else:
            return False
        #end if
        return PesSampler._setup(self)
    #end def

    # ie displacement of positions
    def _shifted(self):
        # check windows
        if self.windows is None or not len(self.windows) == self.D or self.M is None:
            return False
        #end if
        # check noises
        if self.noisy and self.noises is None:
            return False
        #end if
        return PesSampler._shifted(self)
    #end def

    # def _generated(self): not overridden
    def _loaded(self):
        if not all([ls.loaded for ls in self.ls_list]):
            return False
        #end if
        return PesSampler._loaded(self)
    #end def

    def _analyzed(self):
        if not all([ls.analyzed for ls in self.ls_list]):
            return False
        #end if
        return PesSampler._analyzed(self)
    #end def

    def set_hessian(self, hessian):
        self._avoid_protected()
        if hessian is None:
            return
        elif isinstance(hessian, ndarray):
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
        self.cascade()
    #end def

    def set_structure(self, structure, shift_params = None):
        self._avoid_protected()
        if structure is None:
            return
        #end if
        assert isinstance(structure, ParameterSet), 'Structure must be ParameterSet object'
        self.structure = structure.copy(label = 'eqm')
        if shift_params is not None:
            self.structure.shift_params(shift_params)
        #end if
        self.cascade()
    #end def

    def guess_windows(self, windows, window_frac, **kwargs):
        self._avoid_protected()
        self._require_setup()
        if windows is None:
            windows = self.Lambdas**0.5 * window_frac
            self.windows_frac = window_frac
        #end if
        self.set_windows(windows)
    #end def

    def set_windows(self, windows, **kwargs):
        self._avoid_protected()
        self._require_setup()
        if windows is not None:
            assert windows is not None or len(windows) == self.D, 'length of windows differs from the number of directions'
            self.windows = array(windows)
        #end if
        self.cascade()
        self.reset_ls_list(**kwargs)  # always reset ls_list
    #end def

    def set_noises(self, noises, **kwargs):
        self._avoid_protected()
        self._require_setup()
        if noises is None:
            self.noisy = False
            self.noises = None
        else:
            assert(len(noises) == self.D)
            self.noisy = True
            self.noises = array(noises)
        #end if
        self.cascade()
        self.reset_ls_list(**kwargs)  # always reset ls_list
    #end def

    def reset_ls_list(self, D = None, **kwargs):
        self._avoid_protected()
        self._require_shifted()
        if D is None:
            D = range(self.D)
        #end if
        noises = self.noises if self.noisy else self.D * [None]
        ls_list = []
        for d, window, noise in zip(D, self.windows, noises):
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
        self.ls_list = ls_list
        if self.mode == 'pes':
            self.cascade()
            self.load_results()
        elif self.mode == 'files':
            self.generate_jobs(**kwargs)
        #end if
        self.cascade()
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
            'pes_func': self.pes_func,
            'pes_args': self.pes_args,
            'load_func': self.load_func,
            'load_args': self.load_args,
            'mode': self.mode,
        }
        ls_args.update(**kwargs)
        pls_next = ParallelLineSearch(**ls_args)
        return pls_next
    #end def

    def propagate(self, path = None, protect = True, write = True, **kwargs):
        if not self.status.analyzed:
            return
        #end if
        path = path if path is not None else self.path + '_next/'
        # check if manually providing structure
        if 'structure' in kwargs.keys():
            # TODO assert
            pls_next = self.copy(path = path, **kwargs)
        else:
            pls_next = self.copy(path = path, structure = self.structure_next, **kwargs)
        #end if
        self.status.protected = protect
        if write:
            self.write_to_disk()
        #end if
        return pls_next
    #end def

    def generate_jobs(self, pes_func = None, pes_args = {}, **kwargs):
        self._require_shifted()
        assert self.mode in ['nexus', 'files'], 'Supported only in nexus/files modes; present mode: {}'.format(self.mode)
        pes_args = pes_args if not pes_args == {} else self.pes_args
        pes_func = pes_func if pes_func is not None else self.pes_func
        sigma_min = None if not self.noisy else self.noises.min()
        # TODO: check validity of pes_func
        eqm_jobs = self.ls_list[0].generate_eqm_jobs(pes_func, path = self.path, sigma = sigma_min, **pes_args, **kwargs)
        jobs = eqm_jobs
        for ls in self.ls_list:
            jobs += ls.generate_jobs(pes_func, path = self.path, eqm_jobs = eqm_jobs, **pes_args, **kwargs)
        #end for
        self.status.generated = True
        return jobs
    #end def

    def generate_eqm_jobs(self, pes_func = None, pes_args = {}, **kwargs):
        pes_args = pes_args if not pes_args == {} else self.pes_args
        pes_func = pes_func if pes_func is not None else self.pes_func
        sigma_min = None if not self.noisy else self.noises.min()
        eqm_jobs = self.ls_list[0].generate_eqm_jobs(pes_func, path = self.path, sigma = sigma_min, **pes_args, **kwargs)
        return eqm_jobs
    #end def

    def run_jobs(self, interactive = True, eqm_only = False, **kwargs):
        from nexus import run_project
        if eqm_only:
            jobs = self.generate_eqm_jobs(**kwargs)
        else:
            jobs = self.generate_jobs(**kwargs)
        #end if
        if jobs is None or jobs == []:
            return
        #end if
        if interactive:
            print('About to submit the following new jobs:')
            any_new = False
            for job in jobs:
                if not job.submitted:
                    print('  {}'.format(job.path))
                    any_new = True
                #end if
            #end for
            if any_new:
                if input('proceed? (Y/n) ') in ['n', 'N']:
                    exit()
                #end if
            #end if
        #end if
        run_project(jobs)
    #end def

    # can either load based on analyze_func or by providing values/errors
    def load_results(self, load_func = None, values = None, errors = None, **kwargs):  # load_args
        if self.status.protected:
            return
        #end if
        loaded = True
        if self.mode == 'pes' and self.pes_func is not None:
            load_func = None
            values_ls, errors_ls = [], []
            for ls in self.ls_list:
                res = ls.evaluate_pes(pes_func = self.pes_func, pes_args = self.pes_args)
                values_ls.append(res[1])
                errors_ls.append(res[2])
            #end for
        else:
            load_func = load_func if load_func is not None else self.load_func
            values_ls = values if values is not None else self.D * [None]
            errors_ls = errors if errors is not None else self.D * [None]
        #end if
        for ls, values, errors in zip(self.ls_list, values_ls, errors_ls):
            loaded_this = ls.load_results(
                load_func = load_func,
                values = values,
                errors = errors,
                path = self.path,
                **kwargs)
            loaded = loaded and loaded_this
        #end for
        if loaded:
            self.find_eqm_value()
            self.status.generated = True
            self.status.loaded = True
            self.calculate_next()
        #end if
        self.cascade()
    #end def

    def load_eqm_results(self, load_func = None, **kwargs):
        if self.status.protected:
            return
        #end if
        load_func = load_func if load_func is not None else self.load_func
        sigma_min = self.noises.min()
        E, err = self.ls_list[0].load_eqm_results(load_func = load_func, path = self.path, sigma = sigma_min, **kwargs)
        self.structure.value = E
        self.structure.value_err = err
    #end def

    def find_eqm_value(self):
        E, err = None, None
        for ls in self.ls_list:
            for s in ls.structure_list:
                if sum((self.structure.params - s.params)**2) < 1e-10:
                    E, err = s.value, s.value_err
                    break
                #end if
            #end for
        #end for
        self.structure.value = E
        self.structure.value_err = err
    #end def

    def calculate_next(self, **kwargs):
        self._avoid_protected()
        self._require_loaded()
        params_next, params_next_err = self.calculate_next_params(**kwargs)
        self.structure_next = self.structure.copy(params = params_next, params_err = params_next_err)
        self.cascade()
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
        self._require_loaded()
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

    def write_to_disk(self, fname = 'data.p', overwrite = False):
        if path.exists(self.path + fname) and not overwrite:
            print('File {} exists. To overwrite, run with overwrite = True'.format(self.path + fname))
            return
        #end if
        makedirs(self.path, exist_ok = True)
        with open(self.path + fname, mode='wb') as f:
            f.write(dumps(self))
        #end with
    #end def

    def plot_error_surfaces(self, **kwargs):
        for ls in self.ls_list:
            ls.plot_error_surface(**kwargs)
        #end for
    #end def

    def __str__(self):
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
