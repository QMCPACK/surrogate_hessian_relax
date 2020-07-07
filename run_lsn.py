#! /usr/bin/env python3

from parameters import *
from matplotlib import pyplot as plt
from nexus import settings,run_project
from surrogate import IterationData,print_structure_shift,print_optimal_parameters

# edit these accordingly
lsn = obj()
from run_ls_n_minus_1 import data_ls,ls_settings  # replace with actual number
ls_settings.set(**lsn)

# automated from here
n = len(data_ls)
data_last = data_ls[n-1]
R_this = data_ls[n-1].R_next
data = IterationData(n=n,**ls_settings)
data.load_R(data_last.R_next,pos_to_params)
data.load_displacements(data_last.disp, data_last.P_lims)

# run jobs
if __name__=='__main__':
    settings(**nx_settings)
    # eqm jobs
    eqm_jobs = data.get_jobs(data.R,data.eqm_path,dmcsteps=data.dmcsteps)
    P_jobs = eqm_jobs
    # ls jobs
    for p,R in enumerate(data.R_shift):
        if not data.ls_paths[p]==data.eqm_path:
            P_jobs += data.get_jobs(R,data.ls_paths[p],dmcsteps=data.dmcsteps,jastrow=eqm_jobs[data.qmc_j_idx])
        #end if
    #end for
    run_project(P_jobs)
#end if

data.load_PES()
# append to list
data_ls.append( data )

if __name__=='__main__':
    print_structure_shift(data.R,data.R_next)
    print_optimal_parameters(data_ls)
    f,ax = plt.subplots()
    data.plot_PES_fits(ax)
    plt.show()
#end if

