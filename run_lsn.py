#! /usr/bin/env python3

from parameters import *
from copy import deepcopy
from numpy import argmin,polyfit,polyval,poly1d,random,linspace
from matplotlib import pyplot as plt
from qmcpack_analyzer import QmcpackAnalyzer
from nexus import settings,run_project
from surrogate import get_min_params
from ls0 import load_parameters,get_shifts,shift_structure,load_ls_PES

# copy and edit file to e.g. run_ls1.py, then referring to run_ls0, and so
#P_lims  = # give displacement FC limits to go
#E_lim   = # give shift energy limit

from run_ls_n_minus_1 import R_ls,P_ls,S_ls,PES_ls,PES_error_ls # replace with actual number
from run_ls_n_minus_1 import R_next as R_this                  # replace by actual number
optimal   = True
iteration = n # replace by actual number
E_lim     = E_lim_ls[iteration]

# load necessary stuff
P_this,P_lims  = load_parameters()                     # displacement vectors
S_this         = get_shifts(E_lim,P_lims)              # shift grid
R_shift,paths  = shift_structure(R_this,P_this,S_this) # shifted positions
P,PV_this      = pos_to_params(R_this)
P_num          = len(PV_this)

# run jobs
if __name__=='__main__':
    settings(**ls_settings)
    # eqm jobs
    eqm_jobs = get_eqm_jobs(R_this,iteration,'eqm')
    P_jobs = eqm_jobs
    # ls jobs
    for p,pos in enumerate(R_shift):
        if not paths[p]=='eqm':
            P_jobs += get_ls_jobs(pos,iteration,paths[p],eqm_jobs)
        #end if
    #end for
    run_project(P_jobs)
#end if

PES_this, PES_error_this = load_ls_PES('eqm',paths)
dP_mins,E_mins,pfs       = get_dp_E_mins(P_this,S_this,PES_this)
R_next                   = compute_new_structure(R_this,dP_mins,P_this)

# append to the iteration variables
R_ls.append( R_this )
P_ls.append( P_this )
S_ls.append( S_this )
PES_ls.append( PES_this )
PES_error_ls.append( PES_error_this )

if __name__=='__main__':
    print_structure_shift(R_this,R_next)
    plot_curve_fit(S_this,PES_this,PES_error_this,pfs,dP_mins,E_mins)

    print('Optimal parameters:')
    for p in range(P_num):
        print('#'+str(p))
        print('  n=0: '+str(P_ls[0][p]))
        for ni in range(1,iteration+2):
            print('  n='+str(ni)+': '+str(P_ls[ni][p])+' Delta: '+str(P_ls[ni][p]-P_ls[ni-1][p]))
        #end for
    #end for
    plt.show()
#end if

