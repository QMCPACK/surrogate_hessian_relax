class PesResult:
    value = None
    error = None

    def __init__(self, value, error):
        self.value = value
        self.error = error
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
