#! /usr/bin/env python3

from parameters import *
from numpy import argmin

try:
    from run_phonon import R_relax,FC_real,FC_param,FC_e,FC_v,P_orig,P_val,P_opt,num_params
except:
    print('No FC data available: run PHonon calculation first!')
    exit()
#end try

n=0 # zeroth iteration
# choose where to begin
try:
    R_ls = [R_ls_init]
    print('Starting from given geometry:')
except:
    R_ls = [deepcopy(R_relax)]
    print('Starting from relaxed geometry:')
#end try
print(R_ls[0].reshape(shp2))
P,Pv = pos_to_params()
P_ls = [Pv]

settings(**main_settings)


# figure out shifts for this iteration
S_opt = []
for p in range(num_params):
    P_lim = (2*E_lim_ls[n]/FC_e[p])**0.5
    S_opt.append(linspace(-P_lim,P_lim,E_dim))
#end for

# get jobs for this iteration
P_jobs = []
for p in range(num_params):
    for s,shift in enumerate(S_opt[p]):
        pos     = deepcopy(R_ls[n]) + shift*P_opt[p,:]
        pstr    = 'p'+str(p)+'_s'+str(s)
        P_jobs += get_main_job(pos,ls_n,pstr)
    #end for
#end for

if __name__=='__main__':
    run_project(P_jobs)
#end if

# try to load an analyze
try:
    E_load   = []
    Err_load = []
    for j,job in enumerate(P_jobs):
        if job.identifier=='dmc':
           AI      = job.load_analyzer_image()
           E_mean  = AI.qmc[1].scalars.LocalEnergy.mean
           E_error = AI.qmc[1].scalars.LocalEnergy.error
           E_load.append(E_mean)
           Err_load.append(Err_jobs)
        #end if
    #end for
    PES_ls = [array(E_jobs).reshape((num_params,E_dim))]
except:
    print('Could not load energies. Make sure to finish LS calculation first')
#end if

def get_min_params(shifts,PES,n=2):
    pf = polyfit(shifts,PES,n)
    dr = linspace(min(shifts),max(shifts),1001) # more elegant solution, please
    Emin = min(polyval(pf,dr))
    pmin = dr[argmin(polyval(pf,dr))]
    return Emin,pmin
#end def

Emins = []
pmins = []
R_new = deepcopy(R_ls[n])
print('Optimal parameters:')
for p in range(num_params):
    print('#'+str(p))
    Emin,pmin = get_min_params(S_opt[p],PES_ls[n][p,:])
    R_new += pmin*P_opt[p,:] # shift to optimum
    Emins.append(Emin)
    pmins.append(pmin)
#end for
R_ls.append( R_new )
print('')
print('New geometry:')
print(R_new.reshape(shp2))

print(Emin,pmin)
