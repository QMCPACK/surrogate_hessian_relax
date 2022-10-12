#!/usr/bin/env python3

from numpy import array, linalg, diag

from lib.util import Ry, Hartree, Bohr
from lib.parameters import ParameterSet


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
        assert isinstance(structure, ParameterSet), 'Structure must be ParameterSet object'
        self.structure = structure
    #end def

    def init_hessian_structure(self, structure):
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

#end class
