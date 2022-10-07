#!/usr/bin/env python3

# First, the user must set up Nexus according to their computing environment.
from nexus import obj

# Modify the below variables as needed
cores      = 8
presub     = ''
qeapp      = 'pw.x'
p2qapp     = 'pw2qmcpack.x'
qmcapp     = 'qmcpack'
scfjob     = obj(app=qeapp, cores=cores,ppn=cores,presub=presub)
p2qjob     = obj(app=p2qapp,cores=1,ppn=1,presub=presub)
optjob     = obj(app=qmcapp,cores=cores,ppn=cores,presub=presub)
dmcjob     = obj(app=qmcapp,cores=cores,ppn=cores,presub=presub)
nx_settings = obj(
    sleep         = 3,
    pseudo_dir    = 'pseudos',
    runs          = '',
    results       = '',
    status_only   = 0,
    generate_only = 0,
    machine       = 'ws8',
    )
from surrogate_macros import init_nexus
init_nexus(**nx_settings) # initiate nexus
