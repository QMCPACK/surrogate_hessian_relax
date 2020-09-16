#!/usr/bin/env python3
#
# File to load force-constant matrix FC and compute optimal search directions
# based on parameters 

from parameters import *
from numpy import linalg,reshape
from surrogate import load_gamma_k

hessian_file = '../phonon/FC.fc'    # default FC file

try:
    from run_relax import pos_relax
    from run_relax import cell_relax
    print('Loaded pos_relax from run_relax.')
except:
    print('Could not get pos_relax. Run relax first!')
    exit()
#end try

#if __name__=='__main__':
    # run phonon calculation
#end if

hessian_pos  = load_gamma_k(hessian_file,num_prt)
if relax_cell:
    hessian_real = get_hessian_for_cell(hessian_pos) # needs implementation in parameters.py
else:
    hessian_real = hessian_pos
#end if
hessian_delta = delta.T @ hessian_real @ delta

# optimal search directions
P           = delta.shape[1]
Lambda,U    = linalg.eig(hessian_delta)
U           = U.T
directions  = delta @ U.T

# print output
if __name__=='__main__':
    #print_fc_matrix(fc=FC,num_prt=num_prt)

    print('Displacement vector representation of parameters: (Delta)')
    for p in range(P):
        print('#'+str(p))
        print(reshape(delta[:,p],(-1,dim)).round(3))
    #end for
    print('')

    print('Parameters Hessian (H_Delta)')
    print(hessian_delta.round(3))
    print('')
    print('Eigenvectors (U):')
    print(U.round(3))
    print('')
    print('Eigenvalues (Lambda):')
    print(Lambda.round(3))
    print('')

    print('Optimal search directions (Delta @ U.T):')
    for d in range(P):
        print('#'+str(d))
        print(reshape(directions[:,d],(-1,dim)).round(3))
    #end for
    print('')
#end if
