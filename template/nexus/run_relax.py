#! /usr/bin/env python3

from nexus import settings,run_project,obj

from parameters import nx_settings,pos_to_params,params_to_pos,dim,pos_init,get_relax_job,elem
from surrogate_tools import print_relax,get_relax_structure

relax_dir = '../relax'
relax_settings = obj(
    path       = relax_dir,
    suffix     = 'relax.in',
    relax_cell = False,
    pos_units  = 'B',
    )

if __name__=='__main__':
    settings(**nx_settings)
    relax = get_relax_job(pos_init,relax_dir)
    run_project(relax)
#end if

pos_relax    = get_relax_structure(**relax_settings)
params_relax = pos_to_params(pos_relax)

if __name__=='__main__':
    print_relax(elem,pos_relax,params_relax,dim)
#end if
