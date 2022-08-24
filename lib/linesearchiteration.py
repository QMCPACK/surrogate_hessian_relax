#!/usr/bin/env python3

from numpy import array
from dill import loads

from lib.parameters import ParameterStructure
from lib.hessian import ParameterHessian
from lib.parallellinesearch import ParallelLineSearch


# load pickle from disk
def load_from_disk(path):
    try:
        with open(path, mode='rb') as f:
            data = loads(f.read())
        #end with
        return data
    except FileNotFoundError:
        return None
    #end try
#end def


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
        if len(self.pls_list) == 1:
            return self.pls_list[0]
        else:
            return self.pls_list[-1]
        #end if
    #end def

    def pls(self, i = None):
        if i is None:
            return self._get_current_pls()
        elif i < len(self.pls_list):
            return self.pls_list[i]
        else:
            return None
        #end if
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
                print('Loaded pls{} from {}'.format(i, path))
                i += 1
            else:
                print('Could not find pls{} from {}'.format(i, path))
                load_failed = True
            #end if
        #end while
        self.pls_list = pls_list
    #end def

    def _load_linesearch_pickle(self, path):
        return load_from_disk(path)
    #end def

    def propagate(self, **kwargs):
        pls_next = self._get_current_pls().propagate(path = self._get_pls_path(len(self.pls_list)), **kwargs)
        self.pls_list.append(pls_next)
    #end

    def get_params(self, p = None, get_errs = True):
        params = [list(self.pls(0).get_params())]
        params_err = [list(self.pls(0).get_params_err())]
        for pls in self.pls_list:
            if pls.calculated:
                params.append(list(pls.structure_next.params))
                params_err.append(list(pls.structure_next.params_err))
            #end if
        #end for
        if p is not None:
            params = array(params)[:, p]
            params_err = array(params_err)[:, p]
        else:
            params = array(params)
            params_err = array(params_err)
        #end if
        if get_errs:
            return params, params_err
        else:
            return params
        #end if
    #end def

    def __repr__(self):
        string = self.__class__.__name__
        for p, pls in enumerate(self.pls_list):
            string += '\n  #{} '.format(p)
        #end for
        # TODO
        return string
    #end def

#end class
