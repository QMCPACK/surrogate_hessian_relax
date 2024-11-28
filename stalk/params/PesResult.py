from numpy import nan, random


class PesResult:
    '''Represents a PES evaluation result as value+error pair (float/nan)'''

    value = None
    error = None

    def __init__(self, value, error=0.0):
        if not isinstance(value, float):
            self.value = nan
            self.error = 0.0
        else:
            self.value = value
            if isinstance(error, float):
                self.error = error
            else:
                self.error = 0.0
            # end if
        # end if
    # end def

    def get_value(self):
        return self.value
    # end def

    def get_error(self):
        return self.error
    # end def

    def get_result(self):
        return self.get_value(), self.get_error()
    # end def

    def add_sigma(self, sigma):
        '''Add artificial white noise to the result for error resampling purposes.'''
        if sigma is not None and sigma > 0.0:
            self.error = (self.error**2 + sigma**2)**0.5
            self.value = sigma * random.randn(1)[0]
        # end if
    # end def

# end class
