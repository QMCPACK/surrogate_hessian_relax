#! /usr/bin/env python3

from parameters import *
from pwscf_analyzer import PwscfAnalyzer
from nexus import settings,run_project
from numpy import diagonal

relax_path = '../relax/relax.in' # default path

def get_relax_structure(relax_path,relax_cell=False):
    try:
        relax_analyzer = PwscfAnalyzer(relax_path)
        relax_analyzer.analyze()
    except:
        print('No relax geometry available: run relaxation first!')
    #end try
    eq_structure = relax_analyzer.structures[len(relax_analyzer.structures)-1]
    R_relax      = eq_structure.positions.reshape(shp1)
    C_relax      = diagonal(relax_analyzer.input.cell_parameters.vectors)
    if relax_cell:
        R_relax += C_relax
    #end if
    return R_relax
#end def

if __name__=='__main__':
    settings(**relax_settings)
    if relax_cell:
        relax = get_relax_job(pos_init,'../relax',cell=cell_init)[0]
    else:
        relax = get_relax_job(pos_init,'../relax')[0]
    #end if
    run_project(relax)
#end if

R_relax = get_relax_structure(relax_path)

if __name__=='__main__':
    print('Relaxed geometry:')
    print(R_relax.reshape(shp2))
#end if
