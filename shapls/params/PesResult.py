from numpy import nan


class PesResult:
    '''Represents a PES evaluation result as value+error pair (float/nan)'''

    value = None
    error = None

    def __init__(self, value, error=0.0):
        if not isinstance(value, float):
            self.value = nan
            self.error = nan
        else:
            self.value = value
            if isinstance(error, float):
                self.error = error
            else:
                self.error = nan
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

# end class
