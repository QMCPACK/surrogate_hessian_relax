#!/usr/bin/env python3
#
# File to load force-constant matrix FC and compute optimal search directions
# based on parameters 

from numpy import linalg,reshape,loadtxt,savetxt
from nexus import settings,obj,run_project

from surrogate_tools import load_gamma_k,compute_hessian,print_hessian_delta

from parameters import pos_to_params,params_to_pos,num_prt,phonon_dir,relax_cell,dim,nx_settings,hessian_file
from run_relax import pos_relax

phonon_dir  = '../phonon'
phonon_file = 'FC.fc'
hessian_settings = obj(
    pos_to_params = pos_to_params,
    params_to_pos = params_to_pos,
    jax_hessian   = False,
    )

# TODO: automate phonon calculations
#if __name__=='__main__':
    #from parameters import get_phonon_jobs
    #settings(**nx_settings)
    #phonon = get_phonon_jobs(pos_relax,phonon_dir)
    #run_project(phonon)
#end if

hessian_pos  = load_gamma_k('{}/{}'.format(phonon_dir,phonon_file),num_prt)
try:
    from parameters import get_hessian_for_cell
    hessian_posc,posc = get_hessian_for_cell(hessian_pos,pos_relax) # needs implementation in parameters.py
except:
    hessian_posc,posc = hessian_pos,pos_relax
#end if

try:
    hessian_delta = loadtxt(hessian_file)
    print('\nLoaded parameter Hessian from file: {}'.format(hessian_file))
except:
    hessian_delta = compute_hessian(
        pos           = posc,
        hessian_pos   = hessian_posc,
        **hessian_settings
        )
    savetxt(hessian_file,hessian_delta)
#end try

# optimal search directions
Lambda,U    = linalg.eig(hessian_delta)
U           = U.T
P           = U.shape[0]

# print output
if __name__=='__main__':
    print_hessian_delta(hessian_delta,U,Lambda)
#end if
