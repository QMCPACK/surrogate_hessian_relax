
class Parameter():
    """Base class for representing an optimizable parameter"""
    kind = None
    value = None
    error = None
    label = None
    unit = None

    def __init__(
        self,
        value,
        error=0.0,
        kind='',
        label='p',
        unit='',
    ):
        self.value = value
        self.error = error
        self.kind = kind
        self.label = label
        self.unit = unit
    # end def

    @property
    def param_err(self):
        return 0.0 if self.error is None else self.error
    # end def

    def print_value(self):
        if self.error is None:
            print('{:<8.6f}             '.format(self.value))
        else:
            print('{:<8.6f} +/- {:<8.6f}'.format(self.value, self.error))
        # end if
    # end def

    def __str__(self):
        return '{:>10s}: {:<10f} {:6s} ({})'.format(self.label, self.value, self.unit, self.kind)
    # end def
# end class
