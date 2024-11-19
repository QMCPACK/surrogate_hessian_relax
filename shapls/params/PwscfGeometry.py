from .ParameterStructure import ParameterStructure
from .ParameterLoader import ParameterLoader


class PwscfGeometry(ParameterLoader):

    def load(self, path, structure, suffix='relax.in'):
        from nexus import PwscfAnalyzer
        assert isinstance(structure, ParameterStructure), "Structure must be a ParameterStructure object."
        ai = PwscfAnalyzer('{}/{}'.format(path, suffix))
        pos = ai.structures[len(ai.structures) - 1].positions
        try:
            axes = ai.structures[len(ai.structures) - 1].axes
        except AttributeError:
            # In case axes is not present in the relaxation
            axes = None
        # end try
        structure.set_position(pos, axes=axes)
    # end def

# end class
