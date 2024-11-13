#!/usr/bin/env python3
'''LineSearchIteration class for treating iteration of subsequent parallel linesearches
'''

from numpy import array, mean

from shapls.util import directorize
from shapls.params import ParameterSet
from shapls.hessian import ParameterHessian
from shapls.pls import ParallelLineSearch
from .util import plot_parameter_convergence, plot_energy_convergence, plot_bundled_convergence, load_from_disk

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Class for line-search iteration
class LineSearchIteration():

    pls_list = []  # list of ParallelLineSearch objects
    path = ''  # base path
    n_max = None  # TODO

    def __init__(
        self,
        path='',
        surrogate=None,
        structure=None,
        hessian=None,
        load=True,
        n_max=0,  # no limit
        # e.g. windows, noises, targets, units, pes_func, pes_args, load_func, load_args ...
        **kwargs,
    ):
        self.path = directorize(path)
        self.pls_list = []
        # try to load pickles
        if load:
            self.load_pls()
        # end if
        if len(self.pls_list) == 0:  # if no iterations loaded, try to initialize
            # try to load from surrogate ParallelLineSearch object
            if surrogate is not None:
                self.init_from_surrogate(surrogate=surrogate, **kwargs)
            # end if
            # when present, manually provided mappings, parameters and positions override those from a surrogate
            if hessian is not None and structure is not None:
                self.init_from_hessian(
                    structure=structure, hessian=hessian, **kwargs)
            # end if
        # end if
    # end def

    def init_from_surrogate(self, surrogate, **kwargs):
        assert isinstance(
            surrogate, ParallelLineSearch), 'Surrogate parameter must be a ParallelLineSearch object'
        pls = surrogate.copy(path=self._get_pls_path(0), **kwargs)
        self.pls_list = [pls]
    # end def

    def init_from_hessian(self, structure, hessian, **kwargs):
        assert isinstance(
            structure, ParameterSet), 'Starting structure must be a ParameterSet object'
        assert isinstance(
            hessian, ParameterHessian), 'Starting hessian must be a ParameterHessian'
        pls = ParallelLineSearch(
            path=self._get_pls_path(0),
            structure=structure,
            hessian=hessian,
            **kwargs
        )
        self.pls_list = [pls]
    # end def

    def _get_pls_path(self, i):
        return '{}pls{}/'.format(self.path, i)
    # end def

    def generate_jobs(self, **kwargs):
        return self._get_current_pls().generate_jobs(**kwargs)
    # end def

    def load_results(self, **kwargs):
        self._get_current_pls().load_results(**kwargs)
    # end def

    def _get_current_pls(self):
        if len(self.pls_list) == 0:
            return None
        if len(self.pls_list) == 1:
            return self.pls_list[0]
        else:
            return self.pls_list[-1]
        # end if
    # end def

    def pls(self, i=None):
        if i is None:
            return self._get_current_pls()
        elif i < len(self.pls_list):
            return self.pls_list[i]
        else:
            return None
        # end if
    # end def

    def load_pls(self):
        pls_list = []
        load_failed = False
        i = 0
        while not load_failed:
            path = '{}data.p'.format(self._get_pls_path(i))
            pls = self._load_linesearch_pickle(path)
            if pls is not None and pls.check_integrity():
                pls_list.append(pls)
                i += 1
            else:
                load_failed = True
            # end if
        # end while
        self.pls_list = pls_list
    # end def

    def _load_linesearch_pickle(self, path):
        return load_from_disk(path)
    # end def

    def propagate(self, i=None, **kwargs):
        if i is not None and i < len(self.pls_list) - 1:
            return
        # end if
        pls_next = self._get_current_pls().propagate(
            path=self._get_pls_path(len(self.pls_list)), **kwargs)
        self.pls_list.append(pls_next)
    # end

    def get_params(self, p=None, get_errs=True):
        params = []
        params_err = []
        for pls in self.pls_list:
            if pls.status.setup:
                params.append(list(pls.structure.params))
                params_err.append(list(pls.structure.params_err))
            # end if
        # end for
        # the last part, not (yet) propagated
        pls = self.pls()
        if pls.status.analyzed:
            params.append(list(pls.structure_next.params))
            params_err.append(list(pls.structure_next.params_err))
        # end if
        if p is not None:
            params = array(params)[:, p]
            params_err = array(params_err)[:, p]
        else:
            params = array(params)
            params_err = array(params_err)
        # end if
        if get_errs:
            return params, params_err
        else:
            return params
        # end if
    # end def

    def get_average_params(self, p=None, get_errs=True, transient=1):
        params, params_err = self.get_params(p=p, get_errs=True)
        params_ave = mean(params[transient:], axis=0)
        params_ave_err = mean(params_err[transient:]**2, axis=0)**0.5
        if get_errs:
            return params_ave, params_ave_err
        else:
            return params_ave
        # end if
    # end def

    def __str__(self):
        string = self.__class__.__name__
        if len(self.pls_list) > 0:
            fmt = '\n  {:<4d} {}    {:<8f} +/- {:<8f}' + \
                self.pls().D * '   {:<8f} +/- {:<8f}'
            fmts = '\n  {:<4s} {}    {:<8s} +/- {:<8s}' + \
                self.pls().D * '   {:<8s} +/- {:<8s}'
            plabels = ['pls', 'status', 'Energy', '']
            for p in range(self.pls().D):
                plabels += ['p' + str(p)]
                plabels += ['']
            # end for
            string += fmts.format(*tuple(plabels))
            for p, pls in enumerate(self.pls_list):
                data = [pls.structure.value, pls.structure.error]
                data[0] = data[0] if not data[0] is None else 0.0
                data[1] = data[1] if not data[1] is None else 0.0
                for param, perr in zip(pls.structure.params, pls.structure.params_err):
                    data.append(param)
                    data.append(perr)
                # end for
                string += fmt.format(p, pls.status.value(),
                                     *tuple(array(data).round(5)))
            # end for
        # end if
        # TODO add parameter and energy printouts
        return string
    # end def

    def pop(self):
        return self.pls_list.pop()
    # end def

    def plot_convergence(
        self,
        transient=1,
        target_convergence=True,
        bundle=True,
        **kwargs
    ):
        if target_convergence:
            targets = self.get_average_params(
                transient=transient, get_errs=False)
        else:
            targets = None
        # end if
        if bundle:
            plot_bundled_convergence(self.pls_list, targets=targets, **kwargs)
        else:
            plot_energy_convergence(self.pls_list, **kwargs)
            plot_parameter_convergence(
                self.pls_list, targets=targets, **kwargs)
    # end def

# end class
