from numpy import array, isscalar, random, diag
from copy import deepcopy

from shapls.util import match_to_tol, get_fraction_error, directorize
from .util import load_xyz, load_xsf
from .ParameterSet import ParameterSet


class ParameterStructureBase(ParameterSet):
    """Base class for representing a mapping between reducible positions (pos, axes) and irreducible parameters."""
    forward_func = None  # mapping function from pos to params
    backward_func = None  # mapping function from params to pos
    forward_args = None  # kwargs for the forward mapping
    backward_args = None  # kwargs for the backward mapping
    pos = None  # real-space position
    axes = None  # cell axes
    dim = None  # dimensionality
    elem = None  # list of elements
    unit = None  # position units
    tol = None  # consistency tolerance
    # TODO: parameter trust intervals
    # FLAGS
    periodic = False  # is the line-search periodic; will axes be supplied to forward_func
    irreducible = False  # is the parameter mapping irreducible
    consistent = False  # are forward and backward mappings consistent

    def __init__(
        self,
        forward=None,  # pos to params
        backward=None,  # params to pos
        pos=None,
        axes=None,
        elem=None,
        params=None,
        params_err=None,
        periodic=False,
        forward_args={},
        backward_args={},
        value=None,
        error=None,
        label=None,
        unit=None,
        dim=3,
        translate=True,  # attempt to translate pos
        tol=1e-7,
        **kwargs,  # kinds, labels, units
    ):
        self.dim = dim
        self.periodic = periodic
        self.label = label
        self.tol = tol
        self.set_forward_func(forward, forward_args)
        self.set_backward_func(backward, backward_args)
        if pos is not None:
            self.set_position(pos, translate=translate)
        # end if
        if axes is not None:
            self.set_axes(axes)
        # end if
        if params is not None:
            self.init_params(params, params_err, **kwargs)
            self.set_params(self.params)
        # end if
        if value is not None:
            self.set_value(value, error)
        # end if
        if elem is not None:
            self.set_elem(elem)
        # end if
    # end def

    # Set forward mapping function and keyword arguments
    def set_forward_func(self, forward_func, forward_args={}):
        self.forward_func = forward_func
        self.forward_args = forward_args
        # Attempt to update params
        new_params = self.map_forward()
        if new_params is not None:
            self.set_params(new_params)
        # end if
        self._check_consistency()
    # end def

    # Set backward mapping function and keyword arguments
    def set_backward_func(self, backward_func, backward_args={}):
        self.backward_func = backward_func
        self.backward_args = backward_args
        # Attempt to update pos+axes
        new_pos, new_axes = self.map_backward()
        if new_pos is not None:
            self.set_position(new_pos, new_axes)
        # end if
        self._check_consistency()
    # end def

    # Set a new pos+axes and, if so configured, translate pos through backward mapping.
    def set_position(self, pos, axes=None, translate=True):
        pos = array(pos)
        assert pos.size % self.dim == 0, f'Position vector inconsistent with {self.dim} dimensions!'
        # Set the new pos+axes
        self.pos = array(pos).reshape(-1, self.dim)
        if axes is not None:
            # Set axes without mapping checks or moves
            self.axes = self.set_axes(axes, check=False)
        # end if

        # If forward_func has been given, update params; if not, return None
        if self.forward_func is not None:
            self.set_params(self.map_forward())
        # end if

        # setting pos will unset value
        self.unset_value()

        # If set up to translate, take another move backward.
        if translate and self.backward_func is not None:
            self.pos, self.axes = self.map_backward()
        # end if
        self._check_consistency()
    # end def

    # Set a new pos+axes and, if so configured, translate pos through backward mapping.
    def set_axes(self, axes, check=True):
        if array(axes).size == self.dim:
            axes = diag(axes)
        else:
            axes = array(axes)
            assert axes.size == self.dim**2, f'Axes vector inconsistent with {self.dim} dimensions!'
            axes = array(axes).reshape(self.dim, self.dim)
        # end if
        # TODO: there's a problem with updating an explicit list of kpoints
        try:
            self.reset_axes(axes)  # use nexus method to get kaxes
        except AttributeError:
            self.axes = axes
        # end try

        # Skip parameter updates and checks if they are expected later
        if check:
            # If forward_func has been given, update params; if not, return None
            self.set_params(self.map_forward(self.pos, self.axes))
            self.unset_value()  # setting axes will unset value
            self._check_consistency()
        # end if
    # end def

    def set_elem(self, elem):
        self.elem = array(elem)
    # end def

    def set_params(self, params, params_err=None):
        ParameterSet.set_params(self, params, params_err)
        # After params have been set, attempt to update pos+axes
        if self.backward_func is not None:
            self.pos, self.axes = self.map_backward()
        # end if
        self._check_consistency()
    # end def

    def set_value(self, value, error=None):
        assert not self.p_list == [
        ] or self.pos is not None, 'Cannot assign value to abstract structure, set params or pos first'
        self.value = value
        self.error = error
    # end def

    def unset_value(self):
        self.value = None
        self.error = None
    # end def

    # Perform forward mapping: if mapping function provided, return new params; else, return None
    def map_forward(self, pos=None, axes=None):
        pos = pos if pos is not None else self.pos
        if self.forward_func is None or pos is None:
            return None
        # end if
        if self.periodic:
            axes = axes if axes is not None else self.axes
            return array(self.forward_func(array(pos), axes), **self.forward_args)
        else:
            return array(self.forward_func(array(pos), **self.forward_args))
        # end if
    # end def

    # Perform backward mapping: if mapping function provided, return new pos+axes; else, return None
    def map_backward(self, params=None):
        params = params if params is not None else self.params
        if self.backward_func is None or params is None:
            return None, None
        # end if
        if self.periodic:
            pos, axes = self.backward_func(array(params), **self.backward_args)
            return array(pos).reshape(-1, 3), array(axes).reshape(-1, 3)
        else:
            return array(self.backward_func(array(params), **self.backward_args)).reshape(-1, 3), None
        # end if
    # end def

    def _check_consistency(self):
        """Check and store consistency status of the present mappings."""
        self.consistent = self.check_consistency()
    # end def

    def check_consistency(self, params=None, pos=None, axes=None, tol=None, verbose=False):
        """Check consistency of present forward-backward mapping.
        If params or pos/axes are supplied, check at the corresponding points. If not, check at the present point.
        """
        if self.forward_func is None or self.backward_func is None:
            return False
        # end if
        tol = tol if tol is not None else self.tol
        axes = axes if axes is not None else self.axes
        if pos is None and params is not None:
            return self._check_params_consistency(array(params), tol)
        elif pos is not None and params is None:
            return self._check_pos_consistency(array(pos), array(axes), tol)
        # end if
        # if both params and pos are given, check their internal consistency
        pos = array(pos) if pos is not None else self.pos
        params = array(params) if params is not None else self.params
        if pos is not None and params is not None:
            params_new = array(self.map_forward(pos, axes))
            pos_new, axes_new = self.map_backward(params)
            if self.periodic:
                return match_to_tol(params, params_new, tol) and match_to_tol(pos, pos_new, tol) and match_to_tol(axes, axes_new, tol)
            else:
                return match_to_tol(params, params_new, tol) and match_to_tol(pos, pos_new, tol)
            # end if
        else:
            return False
        # end if
    # end def

    def _check_pos_consistency(self, pos, axes, tol=None):
        tol = tol if tol is not None else self.tol
        if self.periodic:
            params = self.map_forward(pos, axes)
            pos_new, axes_new = self.map_backward(params)
            consistent = match_to_tol(
                pos, pos_new, tol) and match_to_tol(axes, axes_new, tol)
        else:
            params = self.map_forward(pos, axes)
            pos_new, axes_new = self.map_backward(params)
            consistent = match_to_tol(pos, pos_new, tol)
        # end if
        return consistent
    # end def

    def _check_params_consistency(self, params, tol=None):
        tol = tol if tol is not None else self.tol
        pos, axes = self.map_backward(params)
        params_new = self.map_forward(pos, axes)
        return match_to_tol(params, params_new, tol)
    # end def

    def _shift_pos(self, dpos):
        if isscalar(dpos):
            return self.pos + dpos
        # end if
        dpos = array(dpos)
        assert self.pos.size == dpos.size
        return (self.pos.flatten() + dpos.flatten()).reshape(-1, self.dim)
    # end def

    def shift_pos(self, dpos):
        assert self.pos is not None, 'position has not been set'
        if isscalar(dpos):
            new_pos = self.pos + dpos
        else:
            dpos = array(dpos)
            assert self.pos.size == dpos.size
            new_pos = (self.pos.flatten() + dpos.flatten()
                       ).reshape(-1, self.dim)
        # end if
        self.set_position(new_pos)
    # end def

    def shift_params(self, dparams):
        new_params = ParameterSet.shift_params(self, dparams)
        self.set_params(new_params)
    # end def

    def copy(
        self,
        params=None,
        params_err=None,
        label=None,
        pos=None,
        axes=None,
        **kwargs,
    ):
        structure = deepcopy(self)
        if params is not None:
            structure.set_params(params, params_err)
        # end if
        if pos is not None:
            structure.set_position(pos)
        # end if
        if axes is not None:
            structure.set_axes(axes)
        # end if
        if label is not None:
            structure.label = label
        # end if
        return structure
    # end def

    def pos_difference(self, pos_ref):
        dpos = pos_ref.reshape(-1, 3) - self.pos
        return dpos
    # end def

    def jacobian(self, dp=0.001):
        assert self.consistent, 'The mapping must be consistent'
        jacobian = []
        for p in range(len(self.params)):
            params_this = self.params.copy()
            params_this[p] += dp
            pos, axes = self.map_backward(params_this)
            dpos = self.pos_difference(pos)
            jacobian.append(dpos.flatten() / dp)
        # end for
        return array(jacobian).T
    # end def

    def get_params_distribution(self, N=100):
        return [self.params + self.params_err * g for g in random.randn(N, len(self.p_list))]
    # end def

    def remap_forward(self, forward, N=None, fraction=0.159, **kwargs):
        assert self.consistent, 'The mapping must be consistent'
        params = forward(*self.map_backward(), **kwargs)
        if N is None:
            return params
        elif sum(self.params_err) > 0:  # resample errorbars
            psdata = [forward(*self.map_backward(p))
                      for p in self.get_params_distribution(N=N)]
            params_err = array([get_fraction_error(ps, fraction=fraction)[
                               1] for ps in array(psdata).T])
            return params, params_err
        else:  # errors are zero
            return params, 0 * params
        # end if
    # end def

    def __str__(self):
        string = self.__class__.__name__
        if self.label is not None:
            string += ' ({})'.format(self.label)
        # end if
        if self.forward_func is None:
            string += '\n  forward mapping: not set'
        else:
            string += '\n  forward mapping: {}'.format(self.forward_func)
        # end if
        if self.backward_func is None:
            string += '\n  backward mapping: not set'
        else:
            string += '\n  backward mapping: {}'.format(self.backward_func)
        # end if
        if self.consistent:
            string += '\n  consistent: yes'
        else:
            string += '\n  consistent: no'
        # end if
        # params
        if self.params is None:
            string += '\n  params: not set'
        else:
            string += '\n  params:'
            for param in self.params:
                string += '\n    {:<10f}'.format(param)  # TODO: ParameterBase
            # end for
        # end if
        # pos
        if self.pos is None:
            string += '\n  pos: not set'
        else:
            string += '\n  pos ({:d} atoms)'.format(len(self.pos))
            for elem, pos in zip(self.elem, self.pos):
                string += '\n    {:2s} {:<6f} {:<6f} {:<6f}'.format(
                    elem, pos[0], pos[1], pos[2])
            # end for
        # end if
        if self.periodic:
            string += '\n  periodic: yes'
            if self.axes is None:
                string += '\n  axes: not set'
            else:
                string += '\n  axes:'
                for axes in self.axes:
                    string += '\n    {:<6f} {:<6f} {:<6f}'.format(
                        axes[0], axes[1], axes[2])
                # end for
            # end if
        else:
            string += '\n  periodic: no'
        # end if
        return string
    # end def

    def relax(
        self,
        pes_func=None,
        pes_args={},
        path='relax',
        mode='nexus',
        load_func=None,
        **kwargs,
    ):
        if pes_func is None:
            print('Provide a pes_func!')
            return
        # end if
        if mode == 'nexus':
            jobs = pes_func(self, directorize(path), **pes_args)
            from nexus import run_project
            run_project(jobs)
            if load_func is not None:
                pos, axes = load_func(path)
                self.set_position(pos, axes=axes)
            # end if
        elif mode == 'pes':
            ParameterSet.relax(self, pes_func=pes_func,
                               pes_args=pes_args, **kwargs)
        # end if
    # end def

    def load(
        self,
        path='relax',
        xyz_file=None,
        xsf_file=None,
        load_func=None,
        load_args={},
        c_pos=1.0,
        c_axes=1.0,
        make_consistent=True,
        verbose=True,
        **kwargs,
    ):
        path = directorize(path)
        if load_func is not None:
            pos, axes = load_func(path, **load_args)
        elif xyz_file is not None:
            fname = '{}{}'.format(path, xyz_file)
            pos = load_xyz(fname)
        elif xsf_file is not None:
            fname = '{}{}'.format(path, xsf_file)
            pos, axes = load_xsf(fname)
        else:
            print('Not loaded')
        # end if
        pos *= c_pos
        self.set_position(pos)
        if self.periodic:
            axes *= c_axes
            self.set_axes(axes)
        # end if
        if make_consistent:
            # Map forth and back for idempotency
            self.params = self.map_forward()
            self.pos, self.axes = self.map_backward()
        # end if
        pos_diff = self.pos - pos
        pos_diff -= pos_diff.mean(axis=0)
        if verbose:
            print('Position difference')
            print(pos_diff.reshape(-1, 3))
        # end if
    # end def

# end class
