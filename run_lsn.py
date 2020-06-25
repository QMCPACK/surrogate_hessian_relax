#! /usr/bin/env python3

from parameters import *
from copy import deepcopy
from numpy import argmin,polyfit,polyval,poly1d,random,linspace
from matplotlib import pyplot as plt
from qmcpack_analyzer import QmcpackAnalyzer
from nexus import settings,run_project
from surrogate import get_min_params,get_shifts
from run_ls0 import shift_structure,compute_new_structure,get_dp_E_mins,load_ls_PES,print_optimal_parameters,plot_curve_fit,print_structure_shift

# default values to be overwritten
iteration    = n # replace by actuan number
from run_ls_n_minus_1 import PF_n_ls,E_lim_ls,dmc_steps_ls,prefix,use_optimal        # replace with actual number
PF_n_ls.append(4)
E_lim_ls.append(0.001)
dmc_steps_ls.append(10)
from parameters import *

# load from previous
from run_ls_n_minus_1 import R_ls,P_ls,S_ls,PES_ls,PES_error_ls,PV_ls,E_ls,Err_ls  # replace with actual number
from run_ls_n_minus_1 import R_next as R_this                                        # replace by actual number
from run_ls_n_minus_1 import P_lims,P_this                                           # replace by actual number


# load necessary stuff
E_lim          = E_lim_ls[iteration]
dmc_steps      = dmc_steps_ls[iteration]
PF_n           = PF_n_ls[iteration]

S_this         = get_shifts(E_lim,P_lims,S_num)        # shift grid
R_shift,paths  = shift_structure(R_this,P_this,S_this) # shifted positions
P,PV_this      = pos_to_params(R_this)                 # parameter value
P_num          = len(PV_this)                          # number of parameters

# run jobs
if __name__=='__main__':
    settings(**ls_settings)
    # eqm jobs
    eqm_jobs = get_eqm_jobs(R_this,iteration,'eqm',dmc_steps)
    P_jobs = eqm_jobs
    # ls jobs
    for p,pos in enumerate(R_shift):
        if not paths[p]=='eqm':
            P_jobs += get_ls_jobs(pos,iteration,paths[p],eqm_jobs,dmc_steps)
        #end if
    #end for
    run_project(P_jobs)
#end if

PES_this,PES_error_this,E_eqm,Err_eqm = load_ls_PES('eqm',paths)
dP_mins,E_mins,pfs                    = get_dp_E_mins(P_this,S_this,PES_this,pf_n=PF_n)
R_next                                = compute_new_structure(R_this,dP_mins,P_this)
P,PV_next                             = pos_to_params(R_next)

# append to the iteration variables
R_ls.append( R_this )
P_ls.append( P_this )
S_ls.append( S_this )
PES_ls.append( PES_this )
PES_error_ls.append( PES_error_this )
PV_ls.append(PV_this)
E_ls.append(E_eqm)
Err_ls.append(Err_eqm)

if __name__=='__main__':
    print_structure_shift(R_this,R_next)
    print_optimal_parameters(PV_ls+[PV_next])
    plot_curve_fit(S_this,PES_this,PES_error_this,pfs,dP_mins,E_mins)
    plt.show()
#end if
