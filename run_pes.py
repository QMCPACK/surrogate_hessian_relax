#! /usr/bin/env python3

# calculate brute-force PES from parameters

from parameters import *
from numpy import unravel_index,prod,polyfit,linspace
from nexus import settings,run_project

try:
    from run_phonon import R_relax,FC_param
except:
    print('No relax geometry available: run relaxation first!')
    exit()
#end try


settings(**nx_settings)

P_orig,P_val = pos_to_params(R_relax)
num_params   = len(P_val)
S_orig       = []
for p in range(num_params):
    P_lim = (2*E_lim_pes/FC_param[p,p])**0.5
    S_orig.append(linspace(-P_lim,P_lim,PES_dim))
#end for
S_orig_mesh = meshgrid(*tuple(S_orig))
S_orig_shp  = S_orig_mesh[0].shape

P_jobs = []
for j in range(prod(S_orig_shp)):
    pos  = deepcopy(R_relax)
    ind  = unravel_index(j,S_orig_shp)
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

try:
    # load PES derivatives
    PES_param = []
    for j,job in enumerate(P_jobs):
        E = job.load_analyzer_image().E
        PES_param.append(E)
    #end for
    PES_param = reshape(array(PES_param),S_orig_shp)
except:
    print('Could not load PES. Make sure to run PES calculation first')
    exit()
#end try


PES_pfs = []
PES_dEs = []
slicing = []
for p in range(num_params):
    slicing.append(int(S_orig_shp[p]/2)) # assume middle to be zero
#end for
for p in range(num_params):
    dE = PES_param[get_1d_sli(p,slicing)]
    dp = S_orig[p]
    PES_dEs.append(dE)
    PES_pfs.append(polyfit(dp,dE,2))
#end for

# all pairs of parameters
p22s = []
p33s = []
p44s = []
for p0 in range(num_params):
    for p1 in range(p0+1,num_params):
        X = S_orig_mesh[p0][get_2d_sli(p0,p1,slicing)]+P_val[p0]
        Y = S_orig_mesh[p1][get_2d_sli(p0,p1,slicing)]+P_val[p1]
        PES_fit = PES_param[get_2d_sli(p0,p1,slicing)]
        p22s.append(bipolyfit(X,Y,PES_param,2,2))
        p33s.append(bipolyfit(X,Y,PES_param,3,3))
        p44s.append(bipolyfit(X,Y,PES_param,4,4))
    #end for
#end for

if __name__=='__main__':
    print('Displacement vector representation of parameters:')
    for p in range(num_params):
        print('#'+str(p))
        print(reshape(P_orig[p],shp2))
        print('PES derivative:')
        print(2*PES_pfs[p][0])
    #end for
    print('')
#end if
