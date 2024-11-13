#!/usr/bin/env python3
'''Cascade status of a PES being sampled.
'''

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# Class for managing PesSampler objects
class CascadeStatus():
    setup = False
    shifted = False
    generated = False
    loaded = False
    analyzed = False
    protected = False

    def __init__(self):
        pass
    # end def

    def value(self):
        return self.__str__()
    # end def

    def __str__(self):
        string = ''
        for s in [self.setup, self.shifted, self.generated, self.loaded, self.analyzed, self.protected]:
            string += '1' if s else '0'
        # end for
        return string
    # end def
# end class
