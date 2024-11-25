#!/usr/bin/env python3
'''Generic base class for sampling a PES in iterative batches
'''

from shapls.params import PesFunction
from shapls.io import FilesFunction, FilesLoader, NexusGenerator, PesLoader
from .CascadeStatus import CascadeStatus

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# A class for managing sampling of the PES
class PesSampler():
    mode = None
    path = None
    pes = None
    loader = None
    status = None
    jobs = None
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
        **kwargs  # pes and loader arguments
    ):
        if mode == 'pes':
            self.__init_pes_mode__(**kwargs)
        elif mode == 'files':
            self.__init_files_mode__(**kwargs)
        elif mode == 'nexus':
            self.__init_nexus_mode__(**kwargs)
        else:
            raise ValueError(
                'Must provide an operating mode: "nexus", "files", or "pes"')
        # end if
        self.mode = mode
        self.status = CascadeStatus()
        self.cascade()
    # end def

    def __init_pes_mode__(
        self,
        pes=None,
        pes_func=None,
        pes_args={},
        **kwargs  # loader arguments are ignored
    ):
        # Treat the PES
        if pes is None:
            # Construct from func/args; checks are made in PesFunction class
            pes = PesFunction(pes_func, pes_args)
        else:
            assert isinstance(
                pes, PesFunction), 'The PES must be inherited from PesFunction class.'
        # end if
        self.pes = pes
    # end def

    def __init_files_mode__(
        self,
        pes=None,
        pes_func=None,
        pes_args={},
        loader=None,
        load_func=None,
        load_args={},
    ):
        # Treat the PES (required)
        if pes is None:
            # Construct from func/args; checks are made in FilesFunction class
            pes = FilesFunction(pes_func, pes_args)
        else:
            assert isinstance(
                pes, FilesFunction), 'The PES must be inherited from FilesFunction class.'
        # end if
        self.pes = pes

        # Treat the loader (not required)
        if loader is None:
            if load_func is not None:
                loader = FilesLoader(load_func, load_args)
            # end if
        else:
            assert isinstance(
                loader, FilesLoader), 'The files loader must be inherited from FilesLoader class.'
        # end if
        self.loader = loader
    # end def

    def __init_nexus_mode__(
        self,
        pes=None,
        pes_func=None,
        pes_args={},
        loader=None,
        load_func=None,
        load_args={},
    ):
        # Treat the PES (required)
        if pes is None:
            # Construct from func/args; checks are made in FilesFunction class
            pes = NexusGenerator(pes_func, pes_args)
        else:
            assert isinstance(
                pes, NexusGenerator), 'The PES must be inherited from NexusGenerator class.'
        # end if
        self.pes = pes

        # Treat the loader (not required)
        if loader is None:
            if load_func is not None:
                loader = PesLoader(load_args)
                loader.__load__ = load_func
            # end if
        else:
            assert isinstance(loader, PesLoader), 'The files loader must be inherited from PesLoader class.'
        # end if
        self.loader = loader
    # end def

    # reset = True overrides all, including protected
    def cascade(self, reset=False):
        if reset or not isinstance(self.status, CascadeStatus):
            self.status = CascadeStatus()
        # end if
        if self.status.protected:
            return
        # end if
        # go through all status flags in order
        s = self.status
        if s.setup or self._setup():
            s.setup = True
        else:
            s.setup = False
            return
        # end if
        if s.shifted or self._shifted():
            s.shifted = True
        else:
            s.shifted = False
            return

        if s.generated or self._generated():
            s.generated = True
        else:
            s.generated = False
            return

        if s.loaded or self._loaded():
            s.loaded = True
        else:
            s.loaded = False
            return
        # end if
        if s.analyzed or self._analyzed():
            s.analyzed = True
        else:
            s.analyzed = False
            return
        # end if
    # end def

    # tests for the setup stage
    def _setup(self):
        return True
    # end def

    # tests for whether we have a list of positions
    def _shifted(self):
        return True
    # end def

    # tests for generation of the pes
    def _generated(self):
        if self.pes is None:
            return False
        # end if
        if self.mode == 'pes':  # direct pes mode will generate on-fly, no stop
            return True
        else:  # the nexus/files jobs must be generated and submitted separately
            return False
        # end if
    # end def

    # tests for loading of results
    def _loaded(self):
        return True
    # end def

    # tests for analysis
    def _analyzed(self):
        return True
    # end def

    # Do not protected by default
    def _protected(self):
        return False
    # end def

    def _require_setup(self):
        assert self.status.setup, self.msg_setup
    # end def

    def _require_shifted(self):
        assert self.status.shifted, self.msg_shifted
    # end def

    def _require_generated(self):
        assert self.status.generated, self.msg_generated
    # end def

    def _require_loaded(self):
        assert self.status.loaded, self.msg_loaded
    # end def

    def _require_analyzed(self):
        assert self.status.analyzed, self.msg_analyzed
    # end def

    def _avoid_protected(self):
        assert not self.status.protected, self.msg_a_protected
    # end def

    def get_status(self, refresh=True):
        if refresh:
            self.cascade()
        # end if
        return self.status.value()
    # end def
# end class
