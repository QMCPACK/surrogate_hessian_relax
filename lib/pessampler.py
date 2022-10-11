#!/usr/bin/env python3


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
    #end def

    def value(self):
        return self.__repr__()
    #end def

    def __str__(self):
        string = ''
        for s in [self.setup, self.shifted, self.generated, self.loaded, self.analyzed, self.protected]:
            string += '1' if s else '0'
        #end for
        return string
    #end def
#end class


# A class for managing sampling of the PES in three alternative modes: nexus, files, direct
class PesSampler():
    mode = None
    path = None
    pes_func = None
    pes_args = {}
    load_func = None
    load_args = {}
    status = None
    # error messages / instructions
    msg_setup = 'Setup: required but not done'
    msg_shifted = 'Shifted: required but not done'
    msg_generated = 'Generated: required but not done'
    msg_loaded = 'Loaded: required but not done'
    msg_analyzed = 'Analyzed: required but not done'
    msg_a_protected = 'Proctected: not touching'

    def __init__(
        self,
        mode,
        path = '',
        pes_func = None,
        pes_args = {},
        load_func = None,
        load_args = {},
        **kwargs,
    ):
        assert mode in ['nexus', 'files', 'pes'], 'Must provide operating mode'
        self.status = CascadeStatus()
        self.mode = mode
        self.path = path
        self.pes_func = pes_func
        self.pes_args = pes_args
        self.load_func = load_func
        self.load_args = load_args
        self.cascade()
    #end def

    # reset = True overrides all, including protected
    def cascade(self, reset = False):
        if reset or not isinstance(self.status, CascadeStatus):
            self.status = CascadeStatus()
        #end if
        if self.status.protected:
            return
        #end if
        # go through all status flags in order
        s = self.status
        if s.setup or self._setup():
            s.setup = True
        else:
            s.setup = False
            return
        #end if
        if s.shifted or self._shifted():
            s.shifted = True
        else:
            s.shifted = False
            return
        #end if
        if s.generated or self._generated():
            s.generated = True
        else:
            s.generated = False
            return
        #end if
        if s.loaded or self._loaded():
            s.loaded = True
        else:
            s.loaded = False
            return
        #end if
        if s.analyzed or self._analyzed():
            s.analyzed = True
        else:
            s.analyzed = False
            return
        #end if
    #end def

    # tests for the setup stage
    def _setup(self):
        return True
    #end def

    # tests for whether we have a list of positions
    def _shifted(self):
        return True
    #end def

    # tests for generation of the pes
    def _generated(self):
        if self.pes_func is None:
            return False
        #end if
        if self.mode == 'pes':  # direct pes mode will generate on-fly, no stop
            return True
        else:  # the nexus/files jobs must be generated and submitted separately
            return False
        #end if
    #end def

    # tests for loading of results
    def _loaded(self):
        return True
    #end def

    # tests for analysis
    def _analyzed(self):
        return True
    #end def

    # Do not protected by default
    def _protected(self):
        return False
    #end def

    def _require_setup(self):
        assert self.status.setup, self.msg_setup
    #end def

    def _require_shifted(self):
        assert self.status.shifted, self.msg_shifted
    #end def

    def _require_generated(self):
        assert self.status.generated, self.msg_generated
    #end def

    def _require_loaded(self):
        assert self.status.loaded, self.msg_loaded
    #end def

    def _require_analyzed(self):
        assert self.status.analyzed, self.msg_analyzed
    #end def

    def _avoid_protected(self):
        assert not self.status.protected, self.msg_a_protected
    #end def

    def get_status(self, refresh = True):
        if refresh:
            self.cascade()
        #end if
        return self.status.value()
    #end def
#end class
