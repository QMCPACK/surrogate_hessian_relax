#!/usr/bin/env python3

from parameters import *
from numpy import linalg,reshape
from surrogate import load_gamma_k,load_phonon_modes,K_from_W

fc_file     = '../phonon/FC.fc'
ph_file     = '../phonon/PH.dynG1'

try:
    from run_relax import R_relax
    P_orig,P_val = pos_to_params(R_relax)
    num_params   = len(P_val)
except:
    print('Could not get eq_pos. Run relax first!')
    exit()
#end try

try:
    # load force-constant matrix:
    FC_real   = load_gamma_k(fc_file,num_prt)
    FC_param  = P_orig @ FC_real @ P_orig.T
    FC_e,FC_v = linalg.eig(FC_param)
    # optimal search directions
    P_opt     = FC_v @ P_orig
except:
    print('Could not load PHonon force-constant matrices. Run ph.x and q2r.x first!')
    exit()
#end try

try:
    # load normal modes
    PH_w,PH_v = load_phonon_modes(ph_file,num_prt)
    KW_real   = K_from_W(PH_w,PH_v,masses)
    KW_param  = P_orig @ KW_real @ P_orig.T
    KW_e,KW_v = linalg.eig(KW_param)
except:
    print('Could not load Normal mode representation')
#end try

if __name__=='__main__':
    #print_fc_matrix(fc=FC,num_prt=num_prt)

    print('Displacement vector representation of parameters:')
    for p in range(num_params):
        print('#'+str(p))
        print(reshape(P_orig[p],shp2))
    #end for
    print('')

    print('Parameter matrix (PHonon)')
    print(FC_param)
    print('')
    print('Eigenvectors:')
    print(FC_v)
    print('')

    print('Optimal search directions:')
    for p in range(num_params):
        print('#'+str(p))
        print(reshape(P_opt[p],shp2))
    #end for
    print('')
#end if
