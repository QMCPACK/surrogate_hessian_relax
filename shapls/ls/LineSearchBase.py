#!/usr/bin/env python3
'''Generic classes for 1-dimensional line-searches
'''

from numpy import array, random, polyval, polyder, equal

from shapls.util import get_min_params, get_fraction_error

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Class for line-search along direction in abstract context
class LineSearchBase():

    fraction = None
    fit_kind = None
    func = None
    func_p = None
    grid = None
    values = None
    errors = None
    mask = []
    x0 = None
    x0_err = None
    y0 = None
    y0_err = None
    sgn = 1
    fit = None

    def __init__(
        self,
        grid=None,
        values=None,
        errors=None,
        fraction=0.025,
        sgn=1,
        **kwargs,
    ):
        self.fraction = fraction
        self.set_func(**kwargs)
        self.sgn = sgn
        if grid is not None:
            self.set_grid(grid)
        # end if
        if values is not None:
            self.set_values(grid, values, errors,
                            also_search=(self.grid is not None))
        # end if
    # end def

    def set_func(
        self,
        fit_kind='pf3',
        **kwargs
    ):
        self.func, self.func_p = self._get_func(fit_kind)
        self.fit_kind = fit_kind
    # end def

    def get_func(self, fit_kind=None):
        if fit_kind is None:
            return self.func, self.func_p
        else:
            return self._get_func(fit_kind)
        # end if
    # end def

    def _get_func(self, fit_kind):
        if 'pf' in fit_kind:
            func = self._pf_search
            func_p = int(fit_kind[2:])
        else:
            raise ('Fit kind {} not recognized'.format(fit_kind))
        # end if
        return func, func_p
    # end def

    def set_grid(self, grid):
        assert len(grid) > 2, 'Number of grid points must be greater than 2'
        self.reset()
        self.grid = array(grid)
    # end def

    def set_values(self, grid=None, values=None, errors=None, also_search=True):
        grid = grid if grid is not None else self.grid
        assert values is not None, 'must set values'
        assert len(values) == len(
            grid), 'Number of values does not match the grid'
        self.reset()
        if errors is None:
            self.errors = None
        else:
            self.errors = array(errors)
        # end if
        self.set_grid(grid)
        self.values = array(values)
        self.mask = len(values) * [True]
        if also_search and not all(equal(array(values), None)):
            self.search()
        # end if
    # end def

    def search(self, **kwargs):
        """Perform line-search with the preset values and settings, saving the result to self."""
        assert self.grid is not None and self.values is not None
        errors = self.errors[self.mask] if self.errors is not None else None
        res = self._search_with_error(
            self.grid[self.mask],
            self.values[self.mask],
            errors,
            fit_kind=self.fit_kind,
            fraction=self.fraction,
            sgn=self.sgn,
            **kwargs)
        self.x0 = res[0]
        self.y0 = res[2]
        self.x0_err = res[1]
        self.y0_err = res[3]
        self.fit = res[4]
        self.analyzed = True
    # end def

    def disable_value(self, i):
        assert i < len(self.mask), 'Cannot disable element {} from array of {}'.format(
            i, len(self.mask))
        self.mask[i] = False
    # end def

    def enable_value(self, i):
        assert i < len(self.mask), 'Cannot enable element {} from array of {}'.format(
            i, len(self.mask))
        self.mask[i] = True
    # end def

    def _search(
        self,
        grid,
        values,
        fit_kind=None,
        **kwargs,
    ):
        func, func_p = self.get_func(fit_kind)
        return self._search_one(grid, values, func, func_p, **kwargs)
    # end def

    def _search_one(
        self,
        grid,
        values,
        func,
        func_p=None,
        **kwargs,
    ):
        return func(grid, values, func_p, **kwargs)  # x0, y0, fit
    # end def

    def _search_with_error(
        self,
        grid,
        values,
        errors,
        fraction=None,
        fit_kind=None,
        **kwargs,
    ):
        func, func_p = self.get_func(fit_kind)
        x0, y0, fit = self._search_one(grid, values, func, func_p, **kwargs)
        fraction = fraction if fraction is not None else self.fraction
        # resample for errorbars
        if errors is not None:
            x0s, y0s = self._get_distribution(
                grid, values, errors, func=func, func_p=func_p, **kwargs)
            ave, x0_err = get_fraction_error(x0s - x0, fraction=fraction)
            ave, y0_err = get_fraction_error(y0s - y0, fraction=fraction)
        else:
            x0_err, y0_err = 0.0, 0.0
        # end if
        return x0, x0_err, y0, y0_err, fit
    # end def

    def _pf_search(
        self,
        grid,
        values,
        pfn,
        **kwargs,
    ):
        return get_min_params(grid, values, pfn, **kwargs)
    # end def

    def reset(self):
        self.x0, self.x0_err, self.y0, self.y0_err, self.fit = None, None, None, None, None
    # end def

    def get_x0(self, err=True):
        assert self.x0 is not None, 'x0 must be computed first'
        if err:
            return self.x0, self.x0_err
        else:
            return self.x0
        # end if
    # end def

    def get_y0(self, err=True):
        assert self.y0 is not None, 'y0 must be computed first'
        if err:
            return self.y0, self.y0_err
        else:
            return self.y0
        # end if
    # end def

    def get_hessian(self, x=None):
        x = x if x is not None else self.x0
        if self.fit is None:
            return None
        else:
            return polyval(polyder(polyder(self.fit)), x)
        # end if
    # end def

    def get_force(self, x=None):
        x = x if x is not None else 0.0
        if self.fit is None:
            return None
        else:
            return -polyval(polyder(self.fit), x)
        # end if
    # end def

    def get_distribution(self, grid=None, values=None, errors=None, fit_kind=None, **kwargs):
        grid = grid if grid is not None else self.grid
        values = values if values is not None else self.values
        errors = errors if errors is not None else self.errors
        func, func_p = self.get_func(fit_kind)
        assert errors is not None, 'Cannot produce distribution unless errors are provided'
        return self._get_distribution(grid, values, errors, func=func, func_p=func_p, sgn=self.sgn, **kwargs)
    # end def

    def get_x0_distribution(self, errors=None, N=100, **kwargs):
        if errors is None:
            return array(N * [self.get_x0(err=False)])
        # end if
        return self.get_distribution(errors=errors, **kwargs)[0]
    # end def

    def get_y0_distribution(self, errors=None, N=100, **kwargs):
        if errors is None:
            return array(N * [self.get_y0(err=False)])
        # end if
        return self.get_distribution(errors=errors, **kwargs)[1]
    # end def

    # TODO: refactor to support generic fitting functions
    def val_data(self, xdata):
        if self.fit is None:
            return None
        else:
            return polyval(self.fit, xdata)
        # end def
    # end def

    # must have func, func_p in **kwargs
    def _get_distribution(self, grid, values, errors, Gs=None, N=100, **kwargs):
        if Gs is None:
            Gs = random.randn(N, len(errors))
        # end if
        x0s, y0s, pfs = [], [], []
        for G in Gs:
            x0, y0, pf = self._search_one(grid, values + errors * G, **kwargs)
            x0s.append(x0)
            y0s.append(y0)
            pfs.append(pf)
        # end for
        return array(x0s, dtype=float), array(y0s, dtype=float)
    # end def

    def __str__(self):
        string = self.__class__.__name__
        if self.fit_kind is not None:
            string += '\n  fit_kind: {:s}'.format(self.fit_kind)
        # end if
        string += self.__str_grid__()
        if self.x0 is None:
            string += '\n  x0: not set'
        else:
            x0_err = '' if self.x0_err is None else ' +/- {: <8f}'.format(
                self.x0_err)
            string += '\n  x0: {: <8f} {:s}'.format(self.x0, x0_err)
        # end if
        if self.y0 is None:
            string += '\n  y0: not set'
        else:
            y0_err = '' if self.y0_err is None else ' +/- {: <8f}'.format(
                self.y0_err)
            string += '\n  y0: {: <8f} {:s}'.format(self.y0, y0_err)
        # end if
        return string
    # end def

    # str of grid
    def __str_grid__(self):
        if self.grid is None:
            string = '\n  data: no grid'
        else:
            string = '\n  data:'
            values = self.values if self.values is not None else len(
                self.values) * ['-']
            errors = self.errors if self.errors is not None else len(
                self.values) * ['-']
            string += '\n    {:9s} {:9s} {:9s}'.format(
                'grid', 'value', 'error')
            for g, v, e in zip(self.grid, values, errors):
                string += '\n    {: 8f} {:9s} {:9s}'.format(g, str(v), str(e))
            # end for
        # end if
        return string
    # end def

# end class
