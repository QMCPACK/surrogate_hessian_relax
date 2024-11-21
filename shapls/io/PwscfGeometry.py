from shapls.params import GeometryResult
from .ParameterLoader import ParameterLoader


class PwscfGeometry(ParameterLoader):

    def __load__(self, path, suffix='relax.in', c_pos=1.0, **kwargs):
        from nexus import PwscfAnalyzer
        ai = PwscfAnalyzer('{}/{}'.format(path, suffix))
        ai.analyze()
        pos = ai.structures[len(ai.structures) - 1].positions * c_pos
        try:
            axes = ai.structures[len(ai.structures) - 1].axes
        except AttributeError:
            # In case axes is not present in the relaxation
            axes = None
        # end try
        return GeometryResult(pos, axes)
    # end def

# end class
