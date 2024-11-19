from numpy import array
from scipy.optimize import minimize
from copy import deepcopy

from .Parameter import Parameter
from .PesFunction import PesFunction


class ParameterSet():
    """Base class for representing a set of parameters to optimize"""
    p_list = []  # list of Parameter objects
    value = None  # energy value
    error = None  # errorbar
    label = None  # label for identification

    def __init__(
        self,
        params=None,
        value=None,
        error=None,
        label=None,
        **kwargs,  # params_err, units, kinds
    ):
        self.label = label
        if params is not None:
            self.init_params(params, **kwargs)
        # end if
        if value is not None:
            self.set_value(value, error)
        # end if
    # end def

    def init_params(self, params, params_err=None, units=None, labels=None, kinds=None, **kwargs):
        if params_err is None:
            params_err = len(params) * [params_err]
        else:
            assert len(params_err) == len(params)
        # end if
        if units is None or isinstance(units, str):
            units = len(params) * [units]
        else:
            assert len(units) == len(params)
        # end if
        if kinds is None or isinstance(kinds, str):
            kinds = len(params) * [kinds]
        else:
            assert len(kinds) == len(params)
        # end if
        if labels is None:
            labels = len(params) * [labels]
        else:
            assert len(labels) == len(labels)
        # end if
        p_list = []
        for p, (param, param_err, unit, label, kind) in enumerate(zip(params, params_err, units, labels, kinds)):
            lab = label if label is not None else 'p{}'.format(p)
            parameter = Parameter(
                param, param_err, unit=unit, label=lab, kind=kind)
            p_list.append(parameter)
        # end for
        self.p_list = p_list
    # end def

    def set_params(self, params, params_err=None):
        if params is None:
            return
        # end if
        if self.params is None:
            self.init_params(params, params_err)
        # end if
        if params_err is None:
            params_err = len(params) * [0.0]
        # end if
        for sparam, param, param_err in zip(self.p_list, params, params_err):
            sparam.value = param
            sparam.error = param_err
        # end for
        self.unset_value()
    # end def

    def set_value(self, value, error=None):
        assert self.params is not None, 'Cannot assign value to abstract structure, set params first'
        self.value = value
        self.error = error
    # end def

    def unset_value(self):
        self.value = None
        self.error = None
    # end def

    @property
    def params(self):
        if self.p_list == []:
            return None
        else:
            return array([p.value for p in self.p_list])
        # end if
    # end def

    @property
    def params_err(self):
        if self.p_list == []:
            return None
        else:
            return array([p.param_err for p in self.p_list])
        # end if
    # end def

    def shift_params(self, dparams):
        assert not self.p_list == [], 'params has not been set'
        for p, d in zip(self.p_list, dparams):
            p.value += d
        # end for
        self.unset_value()
    # end def

    def copy(
        self,
        params=None,
        params_err=None,
        label=None,
        **kwargs,
    ):
        structure = deepcopy(self)
        if params is not None:
            structure.set_params(params, params_err)
        # end if
        if label is not None:
            structure.label = label
        # end if
        return structure
    # end def

    def check_consistency(self):
        return True
    # end def

    def relax(
        self,
        pes,
        **kwargs
    ):
        assert isinstance(pes, PesFunction), "Must provide PES as a PesFunction instance."

        # Relax numerically using a wrapper around SciPy minimize
        def relax_aux(p):
            return pes.run(ParameterSet(p))[0]
        # end def
        res = minimize(relax_aux, self.params, **kwargs)
        self.set_params(res.x)
    # end def

# end class
