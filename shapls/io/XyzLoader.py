from numpy import loadtxt, array

from shapls.params.GeometryResult import GeometryResult
from .GeometryLoader import GeometryLoader


class XyzLoader(GeometryLoader):

    def __load__(self, path, suffix='relax.xyz', c_pos=1.0):
        el, x, y, z = loadtxt('{}/{}'.format(path, suffix), dtype=str, unpack=True, skiprows=2)
        pos = array([x, y, z], dtype=float).T * c_pos
        return GeometryResult(pos, axes=None, elem=el)
    # end def

# end class
