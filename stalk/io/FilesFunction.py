class FilesFunction():
    func = None
    args = None

    def __init__(self, func, args={}):
        '''A files PES function is constructed from the files-generating function and arguments.'''
        self.func = func
        self.args = args
    # end def

    def generate(self, structure, path, **kwargs):
        '''Return a list of structure files provided "structure" and "path" arguments.'''
        files = self.func(structure, path, **self.args, **kwargs)
        return files
    # end def

# end class
