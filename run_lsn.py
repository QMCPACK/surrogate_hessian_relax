#! /usr/bin/env python3

from parameters import *
from surrogate import IterationData,surrogate_diagnostics
import pickle

# edit these accordingly
lsn = obj()
from run_ls_n_minus_1 import data_ls,ls_settings  # replace with actual number
ls_settings.set(**lsn)

# automated from here
n = len(data_ls)
data = IterationData(n=n,**ls_settings)

# try to load data
try:
    data   = pickle.load(open(data.path+'data.p',mode='rb'))
    loaded = True
except:
    loaded = False
#end try

if not loaded:
    data_last = data_ls[n-1]
    data.load_R(data_last.R_next,pos_to_params)
    data.load_displacements(data_last.disp, data_last.P_lims)

    from nexus import settings,run_project
    settings(**nx_settings)
    # eqm jobs
    eqm_jobs = data.get_jobs(data.R,data.eqm_path,dmcsteps=data.dmcsteps)
    P_jobs = eqm_jobs
    # ls jobs
    for p,R_shifts in enumerate(data.R_shifts):
        for s,R in enumerate(R_shifts):
            path = data.ls_paths[p][s]
            if not path==data.eqm_path:
                P_jobs += data.get_jobs(R,path,dmcsteps=data.dmcsteps,jastrow=eqm_jobs[data.qmc_j_idx])
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
