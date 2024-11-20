from shapls.params import ParamsResult
from .ParameterLoader import ParameterLoader


class PwscfGeometry(ParameterLoader):

    def load(self, path, suffix='relax.in', **kwargs):
        from nexus import PwscfAnalyzer
        ai = PwscfAnalyzer('{}/{}'.format(path, suffix))
        ai.analyze()
        pos = ai.structures[len(ai.structures) - 1].positions
        try:
            axes = ai.structures[len(ai.structures) - 1].axes
        except AttributeError:
            # In case axes is not present in the relaxation
            axes = None
        # end try
        return ParamsResult(pos, axes)
    # end def

# end class
