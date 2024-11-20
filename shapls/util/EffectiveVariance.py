class EffectiveVariance():
    '''Class to produce relative number of samples to meet an error target.'''

    samples = None
    errorbar = None

    def __init__(
        self,
        samples,
        errorbar
    ):
        assert samples > 0, 'The number of samples must be > 0.'
        self.samples = samples
        assert errorbar > 0, 'The initial errorbar must be > 0.'
        self.errorbar = errorbar
    # end def

    def get_samples(self, errorbar):
        assert errorbar > 0, 'The requested errorbar must be > 0.'
        samples = self.samples * self.errorbar**2 * errorbar**-2
        return max(1, samples)
    # end def

    def get_errorbar(self, samples):
        assert samples > 0, 'The requested sapmles must be > 0'
        return self.errorbar * (float(self.samples) / samples)**0.5
    # end def

# end class
