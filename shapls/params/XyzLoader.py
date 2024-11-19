from numpy import loadtxt, array

from .ParameterStructure import ParameterStructure
from .ParameterLoader import ParameterLoader


class XyzLoader(ParameterLoader):

    def load(self, path, structure, suffix='relax.xyz'):
        assert isinstance(structure, ParameterStructure), "Structure must be a ParameterStructure object."
        el, x, y, z = loadtxt('{}/{}'.format(path, suffix), dtype=str, unpack=True, skiprows=2)
        pos = array([x, y, z], dtype=float).T
        structure.set_elem(el)
        structure.set_position(pos, axes=None)
    # end def

# end class
