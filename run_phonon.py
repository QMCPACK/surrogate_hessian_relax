#!/usr/bin/env python3
#
# File to load force-constant matrix FC and compute optimal search directions
# based on parameters 

from parameters import *
from numpy import linalg,reshape
from surrogate import load_gamma_k,load_phonon_modes,K_from_W

fc_file = '../phonon/FC.fc'    # default FC file
ph_file = '../phonon/PH.dynG1' # default PH file
#R_relax = # give R_relax or load

def load_relax_geometry():
    try:
        R = R_relax
        C = C_relax
    except:
        try:
            from run_relax import R_relax as R
            from run_relax import C_relax as C
            print('Loaded R_relax from run_relax.')
        except:
            print('Could not get R_relax. Run relax first!')
            exit()
        #end try
    #end try
    return R,C
#end def

def load_FC_matrix(fc_file):
    try:
        # load force-constant matrix:
        FC_real = load_gamma_k(fc_file,num_prt)
    except:
        print('Could not load PHonon force-constant matrices from '+fc_file+'. Run ph.x and q2r.x first!')
        exit()
    #end try
    return FC_real
#end def

def load_KW_matrix(ph_file):
    try:
        # load normal modes
        PH_w,PH_v = load_phonon_modes(ph_file,num_prt)
        KW_real   = K_from_W(PH_w,PH_v,masses)
    except:
        print('Could not load Normal mode representation')
        return None
    #end try
    return KW_real
#end def

# get parameters
R_relax,C_relax  = load_relax_geometry()

#if __name__=='__main__':
    # run phonon calculation
#end if

if relax_cell:
    FC_real       = load_FC_matrix(fc_file)
    P_pos,P_val   = pos_to_params(R_relax)
    P_orig,P_val  = pos_to_params_cell(R_relax,C_relax)
    FC_param      = get_fc_for_cell(P_pos @ FC_real @ P_pos.T) # needs implementation in parameters
else:
    P_orig,P_val = pos_to_params(R_relax)
    FC_real      = load_FC_matrix(fc_file)
    FC_param     = P_orig @ FC_real @ P_orig.T
#end if

# optimal search directions
num_params = len(P_val)
FC_e,FC_v  = linalg.eig(FC_param)
P_opt      = FC_v.T @ P_orig
FC_opt     = diag(FC_e)

# get normal mode representation (optional)
if not relax_cell:
    KW_real   = load_KW_matrix(ph_file)
    if not KW_real is None:
        KW_param  = P_orig @ KW_real @ P_orig.T
        KW_e,KW_v = linalg.eig(KW_param)
    #end if
#end if


# print output
if __name__=='__main__':
    #print_fc_matrix(fc=FC,num_prt=num_prt)

    print('Displacement vector representation of parameters:')
    for p in range(num_params):
        print('#'+str(p))
        print(reshape(P_orig[p],(-1,dim)).round(3))
    #end for
    print('')

    print('Parameter matrix (PHonon)')
    print(FC_param.round(3))
    print('')
    print('Eigenvectors:')
    print(FC_v.round(3))
    print('')

    print('Optimal search directions:')
    for p in range(num_params):
        print('#'+str(p))
        print(reshape(P_opt[p],(-1,dim)).round(3))
    #end for
    print('')
#end if
