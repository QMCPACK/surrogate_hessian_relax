#! /usr/bin/env python3

from parameters import *
from numpy import unravel_index,prod

try:
    from calc_relax import eq_pos,P_orig,S_orig,shp_S_orig,S_orig_mesh,num_params
except:
    print('No relax geometry available: run relaxation first!')
    exit()
#end try

settings(**pes_settings)

P_jobs = []
for j in range(prod(shp_S_orig)):
    pos  = deepcopy(eq_pos)
    ind  = unravel_index(j,shp_S_orig)
    pstr = ''
    for p in range(num_params):
        param = P_orig[p,:]
        shift = S_orig_mesh[p][ind]
        pos  += param*shift
        pstr += '_p'+str(ind[p])
    #end for
    P_jobs += get_pes_job(pos,pstr)
#end for

if __name__=='__main__':
    run_project(P_jobs)
#end if
