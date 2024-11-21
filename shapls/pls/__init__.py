"""Surrogate Hessian accelerated parallel line-search: parallel line-search"""

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"

from shapls.pls.ParallelLineSearch import ParallelLineSearch
from shapls.pls.TargetParallelLineSearch import TargetParallelLineSearch
from shapls.pls.PesSampler import PesSampler

__all__ = [
    'ParallelLineSearch',
    'TargetParallelLineSearch',
    'PesSampler'
]
