#!/usr/bin/env python3
#
# File to load force-constant matrix FC and compute optimal search directions
# based on parameters 

from numpy import linalg,reshape,loadtxt,savetxt
from surrogate_tools import load_gamma_k,compute_hessian_jax,compute_hessian_fdiff,print_hessian_delta

from parameters import pos_to_params,params_to_pos,num_prt,phonon_dir,relax_cell,dim,hessian_file,jax_hessian

from run_relax import pos_relax
from run_relax import cell_relax

#if __name__=='__main__':
    # run phonon calculation
#end if

hessian_pos  = load_gamma_k('{}/FC.fc'.format(phonon_dir),num_prt)
if relax_cell:
    hessian_real = get_hessian_for_cell(hessian_pos) # needs implementation in parameters.py
else:
    hessian_real = hessian_pos
#end if

try:
    hessian_delta = loadtxt(hessian_file)
except:
    if jax_hessian: # use JAX to compute hessian numerically
        hessian_delta = compute_hessian_jax(hessian_real,params_to_pos,pos_to_params,pos_relax)
    else: # finite difference: exact for linear mappings
        hessian_delta = compute_hessian_fdiff(hessian_real,params_to_pos,pos_to_params,pos_relax)
    #end if 
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
