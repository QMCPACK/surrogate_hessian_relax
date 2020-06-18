#! /usr/bin/env python3

from parameters import *
from sys import argv

n = 0 # iteration of line search
if len(argv)>1:
    try:
        n = int(argv[1])
    except:
        n = 0
    #end try
#end if

try:
    from calc_relax import eq_pos,P_opts,S_opts,shp_S_opts,S_opts_mesh,num_params,pdim,pval
except:
    print('No relax geometry available: run relaxation first!')
    exit()
#end try

if n==0:
    ls_start = deepcopy(eq_pos)
else:
    from calc_ls_n import get_ls_pos
    ls_start = get_ls_pos(n-1)
    if ls_start==None:
        print('Could not load line search iteration '+str(n-1))
        exit()
    #end if
#end if

settings(**main_settings)

P_jobs = []
ls = 0 # for now
for p in range(num_params):
    param = P_opts[p,:]
    for s,shift in enumerate(S_opts[p]):
        pos     = deepcopy(ls_start)
        pos    += param*shift
        pstr    = 'p'+str(p)+'_s'+str(s)
        P_jobs += get_main_job(pos,ls,pstr)
    #end for
#end for

if __name__=='__main__':
    run_project(P_jobs)
#end if
