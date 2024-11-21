from ..params.GeometryResult import GeometryResult


class ParameterLoader():
    args = None

    def __init__(self, args={}):
        self.args = args
    # end def

    def load(self, path, **kwargs):
        '''The Geometry loader must accept a "path" to input file and return geometry results.
        '''
        args = self.args.copy()
        args.update(kwargs)
        res = self.__load__(path=path, **args)
        if type(res) is tuple:
            # Assume (pos, axes)
            return GeometryResult(res[0], axes=res[1])
        else:
            # Assume only pos
            return GeometryResult(res)
        # end if
    # end def

    def __load__(self, *args, **kwargs):
        raise NotImplementedError(
            "Implement __load__ function in inherited class.")
    # end def

# end class
