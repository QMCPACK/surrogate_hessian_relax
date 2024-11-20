from .PesResult import PesResult


class NexusLoader():
    '''A wrapper class for Nexus PES loader.'''
    func = None
    args = None

    def __init__(self, func, args={}):
        self.func = func
        self.args = args
    # end def

    def load(self, path, **kwargs):
        '''The Nexus PES loader must accept a "path" to input file and return a value/error pair.'''
        value, error = self.func(path=path, **self.args, **kwargs)
        return PesResult(value, error)
    # end def

# end class
