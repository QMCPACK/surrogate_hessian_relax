#! /usr/bin/env python3

from parameters import *
from pwscf_analyzer import PwscfAnalyzer
from nexus import settings,run_project
from numpy import diagonal

relax_path = '../relax/relax.in' # default path

def get_relax_structure(relax_path):
    try:
        relax_analyzer = PwscfAnalyzer(relax_path)
        relax_analyzer.analyze()
    except:
        print('No relax geometry available: run relaxation first!')
    #end try
    eq_structure = relax_analyzer.structures[len(relax_analyzer.structures)-1]
    R_relax      = eq_structure.positions.reshape(shp1)
    if relax_cell:
        C_relax  = eq_structure.axes.diagonal()
        R_relax = (R_relax.reshape(shp2)/C_relax).reshape(shp1)
        return R_relax,C_relax
    else:
        return R_relax,None
    #end if
#end def

if __name__=='__main__':
    settings(**nx_settings)
    if relax_cell:
        relax = get_relax_job(pos_init,'../relax',cell=cell_init)[0]
    else:
        relax = get_relax_job(pos_init,'../relax')[0]
    #end if
    run_project(relax)
#end if

R_relax,C_relax = get_relax_structure(relax_path)

if __name__=='__main__':
    print('Relaxed geometry:')
    print(R_relax.reshape(shp2))
    P,Pv = pos_to_params(R_relax)
    print('Parameter values:')
    for p,pval in enumerate(Pv):
        print(' #'+str(p)+': '+str(pval))
    #end for
    if relax_cell:
        print('Relaxed cell:')
        print(C_relax.reshape((-1,dim)))
    #end if
#end if
