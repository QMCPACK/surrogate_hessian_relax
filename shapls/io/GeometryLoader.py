class GeometryLoader():
    args = None

    def __init__(self, args={}):
        assert isinstance(args, dict), 'Args must be inherited from dictionary.'
        self.args = args
    # end def

    def load(self, path, **kwargs):
        '''The Geometry loader must accept a "path" to input file and return GeometryResult.
        '''
        args = self.args.copy()
        args.update(kwargs)
        res = self.__load__(path=path, **args)
        return res
    # end def

    def __load__(self, *args, **kwargs):
        raise NotImplementedError(
            "Implement __load__ function in inherited class.")
    # end def

# end class
