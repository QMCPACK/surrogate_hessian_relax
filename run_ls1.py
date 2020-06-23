#! /usr/bin/env python3

from parameters import *
from copy import deepcopy
from numpy import argmin,polyfit,polyval,poly1d,random
from matplotlib import pyplot as plt
from qmcpack_analyzer import QmcpackAnalyzer

try:
    from run_phonon import R_relax,FC_real,FC_param,FC_e,FC_v,P_orig,P_val,P_opt,num_params
except:
    print('No FC data available: run PHonon calculation first!')
    exit()
#end try

n = 1
try:
    from run_ls0 import R_ls,P_ls,E_min_ls,P_min_ls,PES_ls,PES_error_ls
    print('Starting from optimal geometry from n='+str(n-1))
except:
    print('No previous geometry available: Start by finishing line search #'+str(n-1))
#end try

# run iteration:
R_this = R_ls[n]
E_lim  = E_lim_ls[n]

# figure out shifts for this iteration
S_opt = []
for p in range(num_params):
    P_lim = (2*E_lim/FC_e[p])**0.5
    S_opt.append(linspace(-P_lim,P_lim,E_dim))
#end for

# define paths and displacements
path_strings = []
shifted_poss = []
for p in range(num_params):
    for s,shift in enumerate(S_opt[p]):
        path_strings.append( 'p'+str(p)+'_s'+str(s) )
        shifted_poss.append( deepcopy(R_this) + shift*P_opt[p,:] )
    #end for
#end for

# run iteration:
if __name__=='__main__':
    settings(**main_settings)
    P_jobs   = []
    for p,pos in enumerate(shifted_poss):
        P_jobs += get_main_job(pos,n,path_strings[p])
    #end for
    run_project(P_jobs)
#end if

# try to load an analyze
try:
    E_load   = []
    Err_load = []
    for s,string in enumerate(path_strings):
        AI = QmcpackAnalyzer('../ls'+str(n)+'/'+string+'/dmc/dmc.in.xml',equilibration=0)
        AI.analyze()
        E_mean  = AI.qmc[1].scalars.LocalEnergy.mean
        E_error = AI.qmc[1].scalars.LocalEnergy.error
        E_load.append(E_mean)
        Err_load.append(E_error)
    #end for
    PES_this       = array(E_load).reshape((num_params,E_dim))
    PES_error_this = array(Err_load).reshape((num_params,E_dim))
except:
    print('Could not load energies. Make sure to finish first n='+str(n))
    exit()
#end if

# analyze LS PES and propose new optimal geometry
R_new = deepcopy(R_this)
f,ax  = plt.subplots()
Emins = []
Pmins = []
for p in range(num_params):
    shift = S_opt[p]
    PES   = PES_this[p,:]
    PESe  = PES_error_this[p,:]
    Emin,Pmin,pf = get_min_params(shift,PES,n=4)
    Emins.append(Emin)
    Pmins.append(Pmin)

    # shift to optimum
    R_new += Pmin*P_opt[p,:]

    # plot PES
    co = random.random((3,))
    s_axis = linspace(min(shift),max(shift))
    ax.errorbar(shift,PES,PESe,linestyle='-',label='p'+str(p),color=co)
    # plot fitted PES
    ax.plot(s_axis,polyval(pf,s_axis),linestyle=':',color=co)
    # plot minima
    ax.plot(Pmin,Emin,'o',label='E='+str(round(Emin,6))+' p='+str(round(Pmin,6)),color=co)
#end for
ax.set_title('Line-search #'+str(n))
ax.set_xlabel('dp')
ax.set_ylabel('E')
ax.legend()

# update values at n+1
P,P_new = pos_to_params(R_new)
R_ls.append(R_new)
P_ls.append(P_new)
# update values at n
E_min_ls.append(Emins)
P_min_ls.append(Pmins)
PES_ls.append(PES_this)
PES_error_ls.append(PES_this)

# print and plot if needed
if __name__=='__main__':
    print('New geometry:')
    print(R_ls[n+1].reshape(shp2))
    print('Shift:')
    print((R_ls[n+1]-R_ls[n]).reshape(shp2))

    print('Optimal parameters:')
    for p in range(num_params):
        print('#'+str(p))
        print('  n=0: '+str(P_ls[0][p]))
        for ni in range(1,n+2):
            print('  n='+str(ni)+': '+str(P_ls[ni][p])+' Delta: '+str(P_ls[ni][p]-P_ls[ni-1][p]))
        #end for
    #end for
    plt.show()
#end if
