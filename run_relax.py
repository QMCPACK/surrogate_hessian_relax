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
    pos_relax    = eq_structure.positions.reshape((-1,))
    if relax_cell:
        cell_relax = eq_structure.axes.diagonal()
        pos_relax  = (pos_relax.reshape((-1,dim))/cell_relax).reshape((-1,))
        return pos_relax,cell_relax
    else:
        return pos_relax,None
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

pos_relax,cell_relax = get_relax_structure(relax_path)

if __name__=='__main__':
    print('Relaxed geometry:')
    print(pos_relax.reshape((-1,dim)))
    try:
        param_vals = delta_pinv @ pos_relax
    except:
        param_vals = pos_to_params(pos_relax)
    #end try
    print('Parameter values:')
    for p,pval in enumerate(param_vals):
        print(' #'+str(p)+': '+str(pval))
    #end for
    if relax_cell:
        print('Relaxed cell:')
        print(cell_relax.reshape((-1,dim)))
    #end if
#end if
