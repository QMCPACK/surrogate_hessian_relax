#!/usr/bin/env python3

from parameters import *
from run_pes import *
from numpy import polyfit

try:
    # load PES derivatives
    PES = []
    for j,job in enumerate(P_jobs):
        E = job.load_analyzer_image().E
        PES.append(E)
    #end for
    PES = reshape(array(PES),shp_S_orig)
    PES_pfs = []
    PES_dEs = []
    slicing = []
    for p in range(num_params):
        slicing.append(int(shp_S_orig[p]/2)) # assume middle to be zero
    #end for
    for p in range(num_params):
        dE = PES[get_1d_sli(p,slicing)]
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
            X = pshift_mesh[p0][get_2d_sli(p0,p1,slicing)]+pval[p0]
            Y = pshift_mesh[p1][get_2d_sli(p0,p1,slicing)]+pval[p1]
            PES_fit = PES[get_2d_sli(p0,p1,slicing)]
            p22s.append(bipolyfit(X,Y,PES,2,2))
            p33s.append(bipolyfit(X,Y,PES,3,3))
            p44s.append(bipolyfit(X,Y,PES,4,4))
        #end for
    #end for
    load_PES = True
except:
    print('Could not load PES')
    load_PES = False
#end try


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
