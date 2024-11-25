from shapls.params.PesResult import PesResult


class PesLoader():
    args = None

    def __init__(self, args={}):
        assert isinstance(args, dict), 'Args must be inherited from dictionary.'
        self.args = args
    # end def

    def load(self, path, sigma=0.0, **kwargs):
        '''The PES loader must accept a "path" to input file and return PesResult.
        '''
        args = self.args.copy()
        args.update(kwargs)
        res = self.__load__(path=path, **args)
        assert isinstance(res, PesResult), 'The __load__ method must return a PesResult instance.'
        # If a non-zero, artificial errorbar is requested, add it to result
        res.add_sigma(sigma)
        return res
    # end def

    def __load__(self, path=None, *args, **kwargs):
        raise NotImplementedError(
            "Implement __load__ function in inherited class.")
    # end def

# end class
