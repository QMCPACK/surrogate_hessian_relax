#!/usr/bin/env python3
"""ParameterHessian class to consider Hessians according to a ParameterSet mapping.
"""

from numpy import array, linalg, diag, isscalar, zeros, ones, where, mean, polyfit

from lib.util import Ry, Hartree, Bohr, directorize, bipolyfit
from lib.parameters import ParameterSet

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


class ParameterHessian():
    """ParameterHessian class to consider Hessians according to a ParameterSet mapping.
    """
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
        """Set the Hessian location as a ParameterSet or derived object"""
        assert isinstance(structure, ParameterSet), 'Structure must be ParameterSet object'
        self.structure = structure
    #end def

    def init_hessian_structure(self, structure):
        """Initialize the Hessian from a structure"""
        assert isinstance(structure, ParameterSet), 'Provided argument is not ParameterSet'
        assert structure.check_consistency(), 'Provided ParameterStructure is incomplete or inconsistent'
        hessian = diag(len(structure.params) * [1.0])
        self._set_hessian(hessian)
        self.hessian_set = False  # this is not an appropriate hessian
    #end def

    def init_hessian_real(self, hessian_real, structure = None):
        structure = structure if structure is not None else self.structure
        assert structure.check_consistency(), 'Provided ParameterStructure is incomplete or inconsistent'
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
            U = array([[1.0]])
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

    def __str__(self):
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

    def compute_fdiff(
        self,
        structure = None,
        dp = 0.01,
        mode = 'pes',
        path = 'fdiff',
        pes_func = None,
        pes_args = {},
        load_func = None,
        load_args = {},
        **kwargs,
    ):
        eqm = structure if structure is not None else self.structure
        P = len(eqm.params)
        dps = array(P * [dp]) if isscalar(dp) else array(dp)
        dp_list, structure_list, label_list = self._get_fdiff_data(eqm, dps)
        if mode == 'pes':
            Es = [pes_func(s, **pes_args)[0] for s in structure_list]
        elif mode == 'nexus':
            from nexus import run_project
            jobs = []
            for s, l in zip(structure_list, label_list):
                dir = '{}{}'.format(directorize(path), l)
                jobs += pes_func(s, dir, **pes_args)
            #end for
            run_project(jobs)
            Es = []
            for l in label_list:
                dir = '{}{}'.format(directorize(path), l)
                E, Err = load_func(path = dir, **load_args)
                Es.append(E)
            #end for
        else:
            raise(AssertionError, 'Mode {} not supported'.format(mode))
        #end if
        Es = array(Es)
        params = eqm.params
        pdiffs = array(dp_list)
        if P == 1:  # for 1-dimensional problems
            pf = polyfit(pdiffs[:, 0], Es, 2)
            hessian = array([[pf[0]]])
        else:
            hessian = zeros((P, P))
            pfs = [[] for p in range(P)]
            for p0, param0 in enumerate(params):
                for p1, param1 in enumerate(params):
                    if p1 <= p0:
                        continue
                    #end if
                    # filter out the values where other parameters were altered
                    ids = ones(len(pdiffs), dtype=bool)
                    for p in range(P):
                        if p == p0 or p == p1:
                            continue
                        #end if
                        ids = ids & (abs(pdiffs[:, p]) < 1e-10)
                    #end for
                    XY = pdiffs[where(ids)]
                    E = array(Es)[where(ids)]
                    X = XY[:, p0]
                    Y = XY[:, p1]
                    pf = bipolyfit(X, Y, E, 2, 2)
                    hessian[p0, p1] = pf[4]
                    hessian[p1, p0] = pf[4]
                    pfs[p0].append(2 * pf[6])
                    pfs[p1].append(2 * pf[2])
                #end for
            #end for
            for p0 in range(P):
                hessian[p0, p0] = mean(pfs[p0])
            #end for
        #end if
        self.init_hessian_array(hessian)
    #end def

    def _get_fdiff_data(self, structure, dps):
        dp_list = [0.0 * dps]
        structure_list = [structure.copy()]
        label_list = ['eqm']
        def shift_params(id_ls, dp_ls):
            dparams = array(len(dps) * [0.0])
            label = 'eqm'
            for p, dp in zip(id_ls, dp_ls):
                dparams[p] += dp
                label += '_p{}'.format(p)
                if dp > 0:
                    label += '+'
                #end if
                label += '{}'.format(dp)
            #end for
            structure_new = structure.copy()
            structure_new.shift_params(dparams)
            structure_list.append(structure_new)
            dp_list.append(dparams)
            label_list.append(label)
        #end def
        for p0, dp0 in enumerate(dps):
            shift_params([p0], [+dp0])
            shift_params([p0], [-dp0])
            for p1, dp1 in enumerate(dps):
                if p1 <= p0:
                    continue
                #end if
                shift_params([p0, p1], [+dp0, +dp1])
                shift_params([p0, p1], [+dp0, -dp1])
                shift_params([p0, p1], [-dp0, +dp1])
                shift_params([p0, p1], [-dp0, -dp1])
            #end for
        #end for
        return dp_list, structure_list, label_list
    #end def

#end class
