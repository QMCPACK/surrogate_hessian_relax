from .NexusLoader import NexusLoader


class PwscfPes(NexusLoader):

    def __init__(self, args={}):
        def load_pwscf(path, suffix='scf.in', **kwargs):
            from nexus import PwscfAnalyzer
            ai = PwscfAnalyzer('{}/{}'.format(path, suffix))
            ai.analyze()
            E = ai.E
            Err = 0.0
            return E, Err
        # end def
        NexusLoader.__init__(self, func=load_pwscf, args=args)
    # end def

# end class
