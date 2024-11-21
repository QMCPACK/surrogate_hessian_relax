class GeometryResult:
    pos = None
    axes = None

    def __init__(self, pos, axes=None):
        self.pos = pos
        self.axes = axes
    # end def

    def get_pos(self):
        return self.pos
    # end def

    def get_axes(self):
        return self.axes
    # end def

    def get_result(self):
        return self.get_pos(), self.get_axes()
    # end def

# end class
