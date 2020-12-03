#! /usr/bin/env python3

from nexus import settings,run_project
from pwscf_analyzer import PwscfAnalyzer

from parameters import relax_dir,nx_settings,pos_to_params,params_to_pos,dim,pos_init,cell_init,relax_cell,get_relax_job,elem
from surrogate_tools import print_relax

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
        relax = get_relax_job(pos_init,relax_dir,cell=cell_init)[0]
    else:
        relax = get_relax_job(pos_init,relax_dir)[0]
    #end if
    run_project(relax)
#end if

pos_relax,cell_relax = get_relax_structure('{}/relax.in'.format(relax_dir))
params_relax         = pos_to_params(pos_relax,cell=cell_relax)

if __name__=='__main__':
    print_relax(elem,pos_relax,params_relax,cell_relax,dim)
#end if
