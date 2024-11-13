from .LineSearch import LineSearch


class LineSearchDummy(LineSearch):

    def __init__(
        self,
        d=0,
        **kwargs,
    ):
        self.d = d
    # end def

    def load_results(self, **kwargs):
        return True
    # end def

    def generate_jobs(self, **kwargs):
        return []
    # end def

    def generate_eqm_jobs(self, **kwargs):
        return []
    # end def

    def evaluate_pes(self, **kwargs):
        return [None, None, None]
    # end def

    def get_x0(self, err=True):
        if err:
            return 0.0, 0.0
        else:
            return 0.0
        # end if
    # end def

# end class
