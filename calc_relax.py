#!/usr/bin/env python3

from parameters import *

try:
    from run_relax import eq_pos
    P_orig,pval = pos_to_params(eq_pos)
    num_params  = P_orig.shape[0]
except:
    print('Could not get eq_pos. Run relax first!')
    exit()
#end try

try:
    # load force-constant matrix:
    FC        = load_gamma_k(fc_file,num_prt)
    FC_params = param_representation(transpose(P_orig),FC)
    FC_e,FC_v = linalg.eig(FC_params)
    FC_opt    = FC_v @ P_orig
    load_FC   = True
except:
    print('Could not load PHonon force-constant matrices. Run ph.x and q2r.x first!')
    load_FC = False
    exit()
#end try

try:
    # load normal modes
    PH_w,PH_v = load_phonon_modes(ph_file,num_prt)
    KW        = K_from_W(PH_w,PH_v,masses)
    KW_params = param_representation(transpose(P_orig),KW)
    KW_e,KW_v = linalg.eig(KW_params)
    KW_opt    = KW_v @ P_orig
    load_KW   = True
except:
    print('Could not load Normal mode representation')
    load_KW = False
#end try

# optimal line search directions and shifts
pdim    = 7
P_opts  = []
S_opts  = []
S_orig  = []
for p in range(num_params):
    # line search
    #P_opts.append(FC_opt.reshape(num_params,shp1))
    P_opts.append(FC_opt[p])
    plim = (2*Elim/FC_e[p])**0.5
    S_opts.append(linspace(-plim,plim,pdim))
    # original
    plim = (2*Elim/FC_params[p,p])**0.5
    S_orig.append(linspace(-plim,plim,pdim))
#end for
P_opts      = array(P_opts)
S_opts      = tuple(S_opts)
S_orig      = tuple(S_orig)
S_orig_mesh = meshgrid(*S_orig)
S_opts_mesh = meshgrid(*S_orig)
shp_S_opts  = S_opts[0].shape
shp_S_orig  = S_orig[0].shape

if __name__=='__main__':
    #print_fc_matrix(fc=FC,num_prt=num_prt)

    print('Displacement vector representation of parameters:')
    for p in range(num_params):
        print('#'+str(p))
        print(reshape(P_orig[p],shp2))
    #end for
    print('')

    print('Parameter matrix (PHonon)')
    print(FC_params)
    print('Eigenvectors:')
    print(FC_v)
    print('')
   
    print('Optimal search directions: (PHonon)')
    for p in range(num_params):
        print('#'+str(p))
        print(reshape(P_opts[p],shp2))
    #end for
    print('')
#end if
