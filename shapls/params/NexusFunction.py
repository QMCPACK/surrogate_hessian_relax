class NexusFunction():
    func = None
    args = None

    def __init__(self, func, args={}):
        '''A Nexus PES function is constructed from the job-generating function and arguments.'''
        self.func = func
        self.args = args
    # end def

    def generate(self, structure, **kwargs):
        jobs = self.func(structure, **self.args, **kwargs)
        return jobs
    # end def

# end class
