from .ParameterStructureBase import ParameterStructureBase


# Class for physical structure (Nexus)
try:
    from structure import Structure

    class ParameterStructure(ParameterStructureBase, Structure):
        kind = 'nexus'

        def __init__(
            self,
            forward=None,
            backward=None,
            params=None,
            **kwargs
        ):
            ParameterStructureBase.__init__(
                self,
                forward=forward,
                backward=backward,
                params=params,
                **kwargs,
            )
            self.to_nexus_structure(**kwargs)
        # end def

        def to_nexus_structure(
            self,
            kshift=(0, 0, 0),
            kgrid=(1, 1, 1),
            units='A',
            **kwargs
        ):
            s_args = {
                'elem': self.elem,
                'pos': self.pos,
                'units': units,
            }
            if self.axes is not None:
                s_args.update({
                    'axes': self.axes,
                    'kshift': kshift,
                    'kgrid': kgrid,
                })
            # end if
            Structure.__init__(self, **s_args)
        # end def

        def to_nexus_only(self):
            self.forward_func = None
            self.backward_func = None
        # end def

    # end class
except ModuleNotFoundError:  # plain implementation if nexus not present
    class ParameterStructure(ParameterStructureBase):
        kind = 'plain'

        def to_nexus_only(self):
            pass
    # end class
# end try
