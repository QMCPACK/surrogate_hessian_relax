#! /usr/bin/env python3

from parameters import *
from surrogate import IterationData,surrogate_diagnostics
from numpy import diagonal
import pickle

n = 0
ls_settings = obj(get_jobs=get_dmc_jobs)

#R_init        = # give R_init or load
#P_init        = # give displacement vectors to go
#P_lims_init   = # give displacement FC limits to go

# automated from here
data_ls = [] # first iteration
data = IterationData(n=n,**ls_settings) # define in parameters.py

# try to load data
try:
    data   = pickle.load(open(data.path+'data.p',mode='rb'))
    loaded = True
except:
    loaded = False
#end try

if not loaded:
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
            data.load_displacements(P_orig,diagonal(FC_param))
        #end if
    #end try

    from nexus import settings,run_project
    settings(**nx_settings)
    # eqm jobs
    noise    = min(data.noises)
    eqm_jobs = data.get_jobs(data.R,data.eqm_path,noise=noise)
    P_jobs   = eqm_jobs
    # ls jobs
    for p,R_shifts in enumerate(data.R_shifts):
        for s,R in enumerate(R_shifts):
            path  = data.ls_paths[p][s]
            noise = data.noises[p]
            if not path==data.eqm_path:
                P_jobs += data.get_jobs(R,path,noise=noise,jastrow=eqm_jobs[data.qmc_j_idx])
            #end if
        #end for
    #end for
    run_project(P_jobs)
    data.load_PES()

    # write data
    data.write_to_file()
#end if

data_ls.append(data)

if __name__=='__main__':
    surrogate_diagnostics(data_ls)
#end if
