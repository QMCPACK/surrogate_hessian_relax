from numpy import array, linspace, concatenate, polyval, sign, equal
from matplotlib import pyplot as plt

from shapls.io.PesLoader import PesLoader
from shapls.util import directorize

from shapls.io import FilesFunction, NexusGenerator
from shapls.params import ParameterSet, PesFunction
from .LineSearchBase import LineSearchBase


# Class for PES line-search in structure context
class LineSearch(LineSearchBase):
    structure = None  # eqm structure
    structure_list = None  # list of LineSearchStructure objects
    jobs_list = None  # boolean list for bookkeeping of jobs
    d = None  # direction count
    W = None
    R = None
    M = None
    Lambda = None
    sigma = None
    # status flags
    shifted = False
    generated = False
    analyzed = False
    loaded = False

    def __init__(
        self,
        structure=None,
        hessian=None,
        d=0,
        sigma=0.0,
        grid=None,
        **kwargs,
    ):
        self.sigma = sigma if sigma is not None else 0.0
        self.d = d
        if structure is not None:
            self.set_structure(structure)
        # end if
        if hessian is not None:
            self.set_hessian(hessian)
            self.figure_out_grid(grid=grid, **kwargs)
            LineSearchBase.__init__(
                self, grid=self.grid, sgn=self.sgn, **kwargs)
            self.shift_structures()  # shift
        else:
            LineSearchBase.__init__(self, **kwargs)
        # end if
    # end def

    def set_structure(self, structure):
        assert isinstance(
            structure, ParameterSet), 'provided structure is not a ParameterSet object'
        assert structure.check_consistency(), 'Provided structure is not a consistent mapping'
        self.structure = structure
    # end def

    def set_hessian(self, hessian):
        self.hessian = hessian
        Lambda = hessian.get_lambda(self.d)
        self.Lambda = abs(Lambda)
        self.sgn = sign(Lambda)
        self.direction = hessian.get_directions(self.d)
    # end def

    def figure_out_grid(self, **kwargs):
        self.grid, self.M = self._figure_out_grid(**kwargs)
    # end def

    def _figure_out_grid(self, M=None, W=None, R=None, grid=None, **kwargs):
        if M is None:
            M = self.M if self.M is not None else 7  # universal default
        # end if
        if grid is not None:
            self.M = len(grid)
        elif R is not None:
            assert not R < 0, 'R cannot be negative, {} requested'.format(R)
            grid = self._make_grid_R(R, M=M)
            self.R = R
        elif W is not None:
            assert not W < 0, 'W cannot be negative, {} requested'.format(W)
            grid = self._make_grid_W(W, M=M)
            self.W = W
        else:
            raise AssertionError('Must characterize grid')
        # end if
        return grid, M
    # end def

    def _make_grid_R(self, R, M):
        R = max(R, 1e-4)
        grid = linspace(-R, R, M)
        return grid
    # end def

    def _make_grid_W(self, W, M):
        R = self._W_to_R(max(W, 1e-4))
        return self._make_grid_R(R, M=M)
    # end def

    def _W_to_R(self, W):
        """Map W to R"""
        R = (2 * W / self.Lambda)**0.5
        return R
    # end def

    def _R_to_W(self, R):
        """Map R to W"""
        W = 0.5 * self.Lambda * R**2
        return W
    # end def

    def shift_structures(self):
        structure_list = []
        jobs_list = []
        for shift in self.grid:
            structure = self._shift_structure(shift)
            structure_list.append(structure)
            jobs_list.append(False)
        # end for
        self.structure_list = structure_list
        self.jobs_list = jobs_list
        self.shifted = True
    # end def

    def add_shift(self, shift):
        structure = self._shift_structure(shift)
        self.structure_list.append(structure)
        self.jobs_list.append(False)
        if shift not in self.grid:
            self.grid = concatenate([self.grid, [shift]])
        # end if
    # end def

    def _shift_structure(self, shift, roundi=4):
        shift_rnd = round(shift, roundi)
        params_this = self.structure.params
        if shift_rnd == 0.0:
            label = 'eqm'
            params = params_this.copy()
        else:
            sgn = '' if shift_rnd < 0 else '+'
            label = 'd{}_{}{}'.format(self.d, sgn, shift_rnd)
            params = params_this + shift * self.direction
        # end if
        structure = self.structure.copy(params=params, label=label)
        return structure
    # end def

    def evaluate_pes(
        self,
        pes_eval
    ):
        '''Evaluate the PES on the line-search grid using an evaluation function.'''
        assert isinstance(
            pes_eval, PesFunction), 'The evaluation function must be inherited from PesFunction class.'
        grid, values, errors = [], [], []
        for shift, structure in zip(self.grid, self.structure_list):
            value, error = pes_eval.evaluate(
                structure, sigma=self.sigma).get_result()
            grid.append(shift)
            values.append(value)
            errors.append(error)
        # end for
        return array(grid), array(values), array(errors)
    # end def

    def generate_ls_jobs(
        self,
        pes_gen,
        path='',
        eqm_jobs=[],
        exclude_eqm=True,
        **kwargs,
    ):
        '''Generate PES jobs on the line-search grid using a job-generating function.'''
        assert self.shifted, 'Must shift parameters first before generating jobs'
        assert isinstance(
            pes_gen, (NexusGenerator, FilesFunction)), 'The evaluation function must be inherited from either NexusGenerator class or FilesFunction class.'
        jobs = []
        for si, structure in enumerate(self.structure_list):
            if self.jobs_list[si]:
                continue
            else:
                self.jobs_list[si] = True
            # end if
            if exclude_eqm and structure.label == 'eqm':
                continue
            # end if
            jobs += pes_gen.generate(structure, self._make_job_path(
                path, structure.label), sigma=self.sigma, eqm_jobs=eqm_jobs)
        # end for
        self.generated = True
        return jobs
    # end def

    # job must accept 0: position, 1: path, 2: sigma
    def generate_eqm_jobs(
        self,
        pes_gen,
        sigma=None,
        path='',
        **kwargs,
    ):
        if self.generated:
            return []
        # end if
        sigma = sigma if sigma is not None else self.sigma
        path = self._make_job_path(path, self.structure.label)
        return pes_gen.generate(self.structure, path, sigma=sigma)
    # end def

    def _make_job_path(self, path, label):
        return '{}{}'.format(directorize(path), label)
    # end def

    def analyze_job(self, structure, loader, path, sigma=None):
        assert isinstance(
            loader, PesLoader), 'The loader function must be inherited from PesLoader class.'
        value, error = loader.load(path=self._make_job_path(
            path, structure.label), sigma=sigma).get_result()
        return value, error
    # end def

    # Loader function
    def analyze_jobs(self, loader, path, sigma=None, prune0=True):
        # TODO: remove sigma argument
        grid, values, errors = [], [], []
        for shift, structure in zip(self.grid, self.structure_list):
            value, error = self.analyze_job(structure, loader, path, sigma=sigma)
            structure.set_value(value, error)
            # FIXME: skipping values messes up the grid <-> list consistency
            if prune0 and value == 0:
                print('ls{}: skipped shift = {}, value {}'.format(
                    self.d, shift, value))
            else:
                grid.append(shift)
                values.append(value)
                errors.append(error)
            # end if
        # end for
        return array(grid), array(values), array(errors)
    # end def

    # Load results directly from 'grid', 'values' and 'errors' args, or else using 'loader'.
    def load_results(self, loader=None, path=None, grid=None, values=None, errors=None):
        grid = grid if grid is not None else self.grid
        if grid is not None and values is not None:
            self.loaded = self.set_results(grid, values, errors)
        else:
            self.loaded = self.set_results(*self.analyze_jobs(loader, path))
        # end if
        return self.loaded
    # end def

    def load_eqm_results(self, loader=None, path=None, values=None, errors=None):
        if values is not None:
            value, error = values, errors
        else:
            value, error = self.analyze_job(self.structure, loader, path)
        # end if
        return value, error
    # end def

    def set_results(
        self,
        grid,
        values,
        errors=None,
        **kwargs
    ):
        if values is None or all(equal(array(values), None)):
            return False
        # end if
        if errors is None:
            errors = 0.0 * array(values)
        # end if
        self.set_values(grid, values, errors, also_search=True)
        self._update_list_values(values, errors)
        return True
    # end def

    def _update_list_values(self, values, errors):
        for s, v, e in zip(self.structure_list, values, errors):
            s.value = v
            s.error = e
        # end for
    # end def

    def get_shifted_params(self):
        return array([structure.params for structure in self.structure_list])
    # end def

    def plot(
        self,
        ax=None,
        figsize=(4, 3),
        color='tab:blue',
        linestyle='-',
        marker='.',
        return_ax=False,
        c_lambda=1.0,  # FIXME: replace with unit conversions
        **kwargs
    ):
        if ax is None:
            f, ax = plt.subplots()
        # end if
        xdata = self.grid
        ydata = self.values
        xmin = xdata.min()
        xmax = xdata.max()
        ymin = ydata.min()
        xlen = xmax - xmin
        xlims = [xmin - xlen / 8, xmax + xlen / 8]
        xllims = [xmin + xlen / 8, xmax - xlen / 8]
        xgrid = linspace(xlims[0], xlims[1], 201)
        xlgrid = linspace(xllims[0], xllims[1], 201)
        ydata = self.values
        edata = self.errors
        x0 = self.x0 if self.x0 is not None else 0
        y0 = self.y0 if self.x0 is not None else ymin
        x0e = self.x0_err
        y0e = self.y0_err
        # plot lambda
        if self.Lambda is not None:
            a = self.sgn * self.Lambda / 2 * c_lambda
            pfl = [a, -2 * a * x0, y0 + a * x0**2]
            stylel_args = {'color': color, 'linestyle': ':'}  # etc
            ax.plot(xlgrid, polyval(pfl, xlgrid), **stylel_args)
        # end if
        # plot the line-search data
        style1_args = {'color': color,
                       'linestyle': 'None', 'marker': marker}  # etc
        style2_args = {'color': color,
                       'linestyle': linestyle, 'marker': 'None'}
        if edata is None or all(equal(array(edata), None)):
            ax.plot(xdata, ydata, **style1_args)
        else:
            ax.errorbar(xdata, ydata, edata, **style1_args)
        # end if
        ax.errorbar(x0, y0, y0e, xerr=x0e, marker='x', color=color)
        ax.plot(xgrid, self.val_data(xgrid), **style2_args)
        if return_ax:
            return ax
        # end if
    # end def

    def __str__(self):
        string = LineSearchBase.__str__(self)
        if self.Lambda is not None:
            string += '\n  Lambda: {:<9f}'.format(self.Lambda)
        # end if
        if self.W is not None:
            string += '\n  W: {:<9f}'.format(self.W)
        # end if
        if self.R is not None:
            string += '\n  R: {:<9f}'.format(self.R)
        # end if
        return string
    # end def

    # str of grid
    def __str_grid__(self):
        if self.grid is None:
            string = '\n  data: no grid'
        else:
            string = '\n  data:'
            values = self.values if self.values is not None else self.M * ['-']
            errors = self.errors if self.errors is not None else self.M * ['-']
            string += '\n    {:11s}  {:9s}  {:9s}  {:9s}'.format(
                'label', 'grid', 'value', 'error')
            for s, g, v, e in zip(self.structure_list, self.grid, values, errors):
                string += '\n    {:11s}  {: 8f}  {:9.9s}  {:<9.9s}'.format(
                    s.label, g, str(v), str(e))
            # end for
        # end if
        return string
    # end def

# end class
