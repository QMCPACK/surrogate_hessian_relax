from stalk.params.PesResult import PesResult
from .PesLoader import PesLoader


class QmcPes(PesLoader):

    def __load__(self, path, qmc_idx=1, suffix='dmc/dmc.in.xml', **kwargs):
        from nexus import QmcpackAnalyzer
        ai = QmcpackAnalyzer('{}/{}'.format(path, suffix), **kwargs)
        ai.analyze()
        LE = ai.qmc[qmc_idx].scalars.LocalEnergy
        E = LE.mean
        Err = LE.error
        return PesResult(E, Err)
    # end def

# end class
