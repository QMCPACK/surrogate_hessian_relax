#!/usr/bin/env python3
'''TargetLineSearch classes for the assessment and evaluation of fitting errors
'''

from numpy import array, argsort
from numpy import equal
from scipy.interpolate import interp1d, PchipInterpolator

from shapls.ls.LineSearchBase import LineSearchBase

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Class for line-search with resampling and bias assessment against target
class TargetLineSearchBase(LineSearchBase):
    target_x0 = None
    target_y0 = None
    target_grid = None
    target_values = None
    bias_mix = None

    def __init__(
        self,
        target_grid=None,
        target_values=None,
        target_y0=None,
        target_x0=0.0,
        bias_mix=0.0,
        **kwargs,
    ):
        LineSearchBase.__init__(self, **kwargs)
        self.target_x0 = target_x0
        self.target_y0 = target_y0
        self.bias_mix = bias_mix
        self.set_target(grid=target_grid, values=target_values, **kwargs)
    # end def

    def set_target(
        self,
        grid,
        values,
        interpolate_kind='cubic',
        **kwargs,
    ):
        if values is None or all(equal(array(values), None)):
            return
        # end if
        sidx = argsort(grid)
        self.target_grid = array(grid)[sidx]
        self.target_values = array(values)[sidx]
        if self.target_y0 is None:
            self.target_y0 = self.target_values.min()  # approximation
        # end if
        self.target_xlim = [grid.min(), grid.max()]
        if interpolate_kind == 'pchip':
            self.target_in = PchipInterpolator(grid, values, extrapolate=False)
        else:
            self.target_in = interp1d(
                grid, values, kind=interpolate_kind, bounds_error=False)
        # end if
    # end def

    def evaluate_target(self, grid):
        assert grid.min() - self.target_xlim[0] > -1e-6 and grid.max(
        ) - self.target_xlim[1] < 1e-6, 'Requested points off the grid: ' + str(grid)
        return self.target_in(0.99999 * grid)
    # end def

    def compute_bias(self, grid=None, bias_mix=None, **kwargs):
        bias_mix = bias_mix if bias_mix is not None else self.bias_mix
        grid = grid if grid is not None else self.target_grid
        return self._compute_bias(grid, bias_mix, **kwargs)
    # end def

    def _compute_xy_bias(self, grid, bias_order=1, **kwargs):
        x0 = 0
        for i in range(bias_order):
            grid_this = array(
                [min([max([p, self.target_grid.min()]), self.target_grid.max()]) for p in (grid + x0)])
            values = self.evaluate_target(grid_this)
            x0, y0, fit = self._search(
                grid_this, values * self.sgn, **kwargs)
        # end for
        bias_x = x0 - self.target_x0
        bias_y = y0 - self.target_y0
        return bias_x, bias_y
    # end def

    def _compute_bias(self, grid, bias_mix=None, **kwargs):
        bias_mix = bias_mix if bias_mix is not None else self.bias_mix
        bias_x, bias_y = self._compute_xy_bias(grid, **kwargs)
        bias_tot = abs(bias_x) + bias_mix * abs(bias_y)
        return bias_x, bias_y, bias_tot
    # end def

    def compute_errorbar(
        self,
        grid=None,
        errors=None,
        **kwargs
    ):
        grid = grid if grid is not None else self.grid
        errors = errors if errors is not None else self.errors
        errorbar_x, errorbar_y = self._compute_errorbar(grid, errors, **kwargs)
        return errorbar_x, errorbar_y
    # end def

    def _compute_errorbar(self, grid, errors, **kwargs):
        values = self.evaluate_target(grid)
        x0, x0_err, y0, y0_err, fit = self._search_with_error(
            grid, values * self.sgn, errors, **kwargs)
        return x0_err, y0_err
    # end def

    def compute_error(
        self,
        grid=None,
        errors=None,
        W=None,
        R=None,
        **kwargs
    ):
        grid = self._figure_out_grid(R=R, W=W, grid=grid)
        bias_x, bias_y, bias_tot = self.compute_bias(grid, **kwargs)
        errorbar_x, errorbar_y = self.compute_errorbar(grid, errors, **kwargs)
        error = bias_tot + errorbar_x
        return error
    # end def

    def _compute_error(self, grid, errors, **kwargs):
        bias_x, bias_y, bias_tot = self._compute_bias(grid, **kwargs)
        errorbar_x, errorbar_y = self._compute_errorbar(grid, errors, **kwargs)
        return bias_tot + errorbar_x
    # end def

    def __str__(self):
        string = LineSearchBase.__str__(self)
        if self.target_grid is not None:
            string += '\n  target grid: set'
        # end if
        if self.target_values is not None:
            string += '\n  target values: set'
        # end if
        string += '\n  bias_mix: {:<4f}'.format(self.bias_mix)
        return string
    # end def

    # str of grid
    # TODO: change; currently overlapping information
    def __str_grid__(self):
        if self.target_grid is None:
            string = '\n  target data: no grid'
        else:
            string = '\n  target data:'
            values = self.target_values if self.target_values is not None else self.M * \
                ['-']
            string += '\n    {:9s} {:9s}'.format('grid', 'value')
            for g, v in zip(self.target_grid, values):
                string += '\n    {: 8f} {:9s}'.format(g, str(v))
            # end for
        # end if
        return string
    # end def


# end class
