class NexusFunction():
    '''A wrapper class for generating Nexus functions to produce and represent a PES.'''
    func = None
    args = None

    def __init__(self, func, args={}):
        '''A Nexus PES function is constructed from the job-generating function and arguments.'''
        self.func = func
        self.args = args
    # end def

    def generate(self, structure, path, **kwargs):
        '''Return a list of Nexus jobs provided "structure" and "path" arguments.'''
        jobs = self.func(structure, path, **self.args, **kwargs)
        return jobs
    # end def

# end class