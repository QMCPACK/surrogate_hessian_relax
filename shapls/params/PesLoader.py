from .PesResult import PesResult


class PesLoader():
    func = None
    args = None

    def __init__(self, func, args={}):
        self.func = func
        self.args = args
    # end def

    # A PES loader must return value and errorbar
    def load(self, structure, path=None, **kwargs):
        value, error = self.func(structure=structure, path=path, **self.args, **kwargs)
        return PesResult(value, error)
    # end def

# end class
