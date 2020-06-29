#! /usr/bin/env python3

from matplotlib import pyplot as plt
from nexus import settings,run_project
from surrogate import IterationData,print_structure_shift,print_optimal_parameters
from parameters import *
from numpy import diagonal

try:
    n    = 0
    data = IterationData(n=n,**ls_settings[n]) # define in parameters.py
except:
    print('Could not init IterationData object. Need a list of iteration data defined in parameters.py')
    exit()
#end try

#R_init        = # give R_init or load
#P_init        = # give displacement vectors to go
#P_lims_init   = # give displacement FC limits to go

# choose R to load
try:
    data.load_R(R_init,pos_to_params)
except:
    try:
        from run_phonon import R_relax
        data.load_R(R_relax,pos_to_params)
        print('Loaded R for n=0 from run_phonon.')
    except:
        print('Could not get R_relax. Run relax first!')
        exit()
    #end try
#end try

# choose displacement parameters to load
try:
    data.load_displacements(P_init,P_lims_init)
except: 
    if data.use_optimal:
        from run_phonon import P_opt
        from run_phonon import FC_opt
        data.load_displacements(P_opt,diagonal(FC_opt))
    else:
        from run_phonon import P_orig
        from run_phonon import FC_param
        data.prefix += 'orig_'
        data.load_displacements(P_orig,diagonal(FC_params))
    #end if
#end try

# run jobs
if __name__=='__main__':
    settings(**nx_settings)
    # eqm jobs
    eqm_jobs = get_eqm_jobs(data.R,data.eqm_path,data.dmc_factor)
    P_jobs = eqm_jobs
    # ls jobs
    for p,R in enumerate(data.R_shift):
        if not data.ls_paths[p]==data.eqm_path:
            P_jobs += get_ls_jobs(R,data.ls_paths[p],eqm_jobs,data.dmc_factor)
        #end if
    #end for
    run_project(P_jobs)
#end if

data.load_PES()
# first item to list
data_ls = [data]

if __name__=='__main__':
    print_structure_shift(data.R,data.R_next)
    print_optimal_parameters(data_ls)
    f,ax = plt.subplots()
    data.plot_PES_fits(ax)
    plt.show()
#end if

