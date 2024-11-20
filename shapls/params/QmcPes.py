from .NexusLoader import NexusLoader


class QmcPes(NexusLoader):

    def __init__(self, args={}):
        def load_qmc(path, qmc_idx=1, suffix='dmc/dmc.in.xml', **kwargs):
            from nexus import QmcpackAnalyzer
            ai = QmcpackAnalyzer('{}/{}'.format(path, suffix))
            ai.analyze()
            LE = ai.qmc[qmc_idx].scalars.LocalEnergy
            E = LE.mean
            Err = LE.error
            return E, Err
        # end def
        NexusLoader.__init__(self, func=load_qmc, args=args)
    # end def

# end class
