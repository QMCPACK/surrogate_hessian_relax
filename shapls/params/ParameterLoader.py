class ParameterLoader():
    args = None

    def __new__(cls, path=None, structure=None, **kwargs):
        '''By default, allow loading from the construction arguments, if path is given.'''
        if path is not None:
            return cls.load(path, structure, **kwargs)
        else:
            return super().__new__(cls)
        # end if
    # end def

    # Dummy loader, by default returns the value and error from structure
    def load(self, path=None, structure=None):
        raise NotImplementedError("Parameter loader not implemented.")
    # end def

# end class
