#! /usr/bin/env python3

from nexus import obj

label        = 'benzene'
axes         = (0,1) # xy
xl,yl        = 'x','y'

# line search settings
ls0 = obj(
    S_num         = 7,
    polyfit_n     = 4,
    E_lim         = 0.01,
    dmc_factor    = 1,
    equilibration = 0,
    path          = '../ls0/',
    )
ls1 = obj(
    S_num         = 7,
    polyfit_n     = 2,
    E_lim         = 0.001,
    dmc_factor    = 16,
    equilibration = 0,
    path          = '../ls1/',
    )

ls_settings = [ls0,ls1]

from GeSe_structure import *
from GeSe_jobs import *
