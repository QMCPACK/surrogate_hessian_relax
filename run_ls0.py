#! /usr/bin/env python3

from copy import deepcopy
from numpy import argmin,polyfit,polyval,poly1d,random,linspace,delete
from matplotlib import pyplot as plt
from qmcpack_analyzer import QmcpackAnalyzer
from nexus import settings,run_project
from surrogate import get_min_params,get_shifts

# default values to be overwritten
iteration    = 0 # don't change this
PF_n_ls      = [4]
E_lim_ls     = [0.01]
dmc_steps_ls = [1]
prefix       = ''
use_optimal  = True

from parameters import *

#R_ls_init     = # give R_init or load
#P_ls_init     = # give displacement vectors to go
#P_lims_init   = # give displacement FC limits to go


# choose R to load
try:
    R_load = R_ls_init
except:
    try:
        from run_phonon import R_relax as R_load
        print('Loaded R_relax from run_phonon.')
    except:
        print('Could not get R_relax. Run relax first!')
        exit()
    #end try
#end try

# choose displacement parameters to load
try:
    P_load      = P_ls_init
    P_lims_load = P_lims_init
except: 
    if use_optimal:
        from run_phonon import P_opt as P_load
        from run_phonon import FC_e as P_lims_load
    else:
        from run_phonon import P_orig as P_load
        from run_phonon import FC_param
        P_lims_load = []
        for p in range(FC_param.shape[0]):
            P_lims_load.append(FC_param[p,p])
        #end for
        prefix += 'orig_'
    #end if
#end try


# define paths and displacements
def shift_structure(R, params, shifts):
    paths = []
    R_shift = []
    for p in range(len(shifts)):
        for s,shift in enumerate(shifts[p]):
            if abs(shift)<1e-10: #eqm
                paths.append( 'eqm' )
            else:
                paths.append( prefix+'p'+str(p)+'_s'+str(s) )
            #end if
            R_shift.append( deepcopy(R) + shift*params[p,:] )
        #end for
    #end for
    return R_shift,paths
#end def

def load_ls_PES(eqm_path,ls_paths,eq=0):
    E_load   = []
    Err_load = []
    # load eqm
    AI = QmcpackAnalyzer('../ls'+str(iteration)+'/'+eqm_path+'/dmc/dmc.in.xml',equilibration=eq)
    AI.analyze()
    E_eqm   = AI.qmc[1].scalars.LocalEnergy.mean
    Err_eqm = AI.qmc[1].scalars.LocalEnergy.error
    # load ls
    for s,path in enumerate(ls_paths):
        AI = QmcpackAnalyzer('../ls'+str(iteration)+'/'+path+'/dmc/dmc.in.xml',equilibration=eq)
        AI.analyze()
        E_mean  = AI.qmc[1].scalars.LocalEnergy.mean
        E_error = AI.qmc[1].scalars.LocalEnergy.error
        E_load.append(E_mean)
        Err_load.append(E_error)
    #end for
    PES_this       = array(E_load).reshape((P_num,E_dim))
    PES_error_this = array(Err_load).reshape((P_num,E_dim))
    return PES_this,PES_error_this,E_eqm,Err_eqm
#end def


def get_dp_E_mins(P,S,PES,pf_n=4):
    dP_mins = []
    E_mins  = []
    pfs     = []
    for s in range(len(S)):
        shift = S[s]
        Emin,Pmin,pf = get_min_params(S[s],PES[s,:],n=pf_n)
        E_mins.append(Emin)
        dP_mins.append(Pmin)
        pfs.append(pf)
    #end for
    return dP_mins,E_mins,pfs
#end def


def compute_new_structure(R,dP,P):
    R_new = deepcopy(R)
    for p in range(len(dP)):
        R_new += dP[p]*P[p,:]
    #end for
    return R_new
#end def


def print_structure_shift(R_old,R_new):
    print('New geometry:')
    print(R_new.reshape(shp2))
    print('Shift:')
    print((R_new-R_old).reshape(shp2))
#end for

def print_optimal_parameters(PV_list):
    print('Optimal parameters:')
    for n in range(len(PV_list)):
        print(' n=:'+str(n) )
        PV_this = PV_list[n]
        if n==0:
            for p in range(len(PV_this)):
                print('  #'+str(p)+': '+str(PV_this[p]))
            #end for
        else:
            PV_last = PV_list[n-1]
            for p in range(len(PV_this)):
                print('  #'+str(p)+': '+str(PV_this[p])+' Delta: '+str(PV_this[p]-PV_last[p]))
            #end for
        #end if
    #end for
#end def

def plot_curve_fit(S,PES,PES_error,pfs,dP_mins,E_mins):
    f,ax = plt.subplots()
    for p in range(len(S)):
        shift = S[p]
        PES   = PES_this[p,:]
        PESe  = PES_error_this[p,:]
        pf    = pfs[p]
        Pmin  = dP_mins[p]
        Emin  = E_mins[p]

        # plot PES
        co = random.random((3,))
        s_axis = linspace(min(shift),max(shift))
        # plot fitted PES
        ax.errorbar(shift,PES,PESe,linestyle='-',label='p'+str(p),color=co)
        ax.plot(s_axis,polyval(pf,s_axis),linestyle=':',color=co)
        ax.plot(Pmin,Emin,'o',label='E='+str(round(Emin,6))+' p='+str(round(Pmin,6)),color=co)
        # plot minima
    #end for
    ax.set_title('Line-search #'+str(iteration))
    ax.set_xlabel('dp')
    ax.set_ylabel('E')
    ax.legend()
#end def

# load necessary stuff
E_lim          = E_lim_ls[iteration]
S_num          = E_dim
dmc_steps      = dmc_steps_ls[iteration]
PF_n           = PF_n_ls[iteration]

R_this         = R_load                                # starting geometry
P_this         = P_load                                # displacement vectors
P_lims         = P_lims_load                           # estimated displacement limits
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

# start iteration variables
R_ls         = [R_this]
P_ls         = [P_this]
S_ls         = [S_this]
PES_ls       = [PES_this]
PES_error_ls = [PES_error_this]
PV_ls        = [PV_this]
E_ls         = [E_eqm]
Err_ls       = [Err_eqm]
# todo: E_min_ls, P_min_ls, etc

if __name__=='__main__':
    print_structure_shift(R_this,R_next)
    print_optimal_parameters(PV_ls+[PV_next])
    plot_curve_fit(S_this,PES_this,PES_error_this,pfs,dP_mins,E_mins)
    plt.show()
#end if
