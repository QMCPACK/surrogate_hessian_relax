"""Surrogate Hessian accelerated parallel line-search: utilities"""

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

from .util import Bohr, Ry, Hartree
from .util import get_min_params, get_fraction_error, match_to_tol, W_to_R, R_to_W
from .util import bipolynomials, bipolyfit, bipolyval
from .util import directorize

__all__ = [Bohr, Ry, Hartree, get_min_params, get_fraction_error, match_to_tol,
           W_to_R, R_to_W, bipolyfit, bipolynomials, bipolyval, directorize]
