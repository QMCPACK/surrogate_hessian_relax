#! /usr/bin/env python3

from nexus import settings,job,run_project,obj
from nexus import generate_pwscf,generate_pw2qmcpack
from nexus import generate_qmcpack,vmc,dmc
from nexus import Structure
from numpy import array,diag,reshape,transpose,mean,linalg,cos,sin,pi,linspace
from surrogate import *

label       = 'label'
fc_file     = '../phonon/FC.fc'
ph_file     = '../phonon/PH.dynG1'
axes        = (0,1) # xy
xl,yl       = 'x','y'
Elim        = 0.01 # Ry
E_lim_pes   = 0.01 # Ry
E_lim_ls    = [0.01,0.001] # Ry
E_dim       = 7
elem      = []
masses    = []

# settings for the structure
a         = 10.0
dim       = 3
pos_init  = array([[]])
num_prt   = len(elem)
valences  = obj( ) # add valences
shp2      = (num_prt,dim)
shp1      = (num_prt*dim)

def generate_structure(pos_vect,a):
    structure = Structure(dim=dim)
    structure.set_axes(axes=diag([a,a,a]))
    structure.add_kmesh(kgrid=(1,1,1),kshift=(0,0,0))
    structure.set_elem(elem)
    structure.pos = reshape(pos_vect,shp2)
    structure.units = 'B'
    return structure
#end def

def pos_to_params(pos):
    params  = [] # unit displacement vectors
    pval    = [] # parameter values
    return array(params),array(pval)
#end def


# setting for the surrogate job
scf_inputs = obj( 
    pseudos     = [],
    input_dft   = 'pbe', 
    ecut        = 300, 
    ecutrho     = 2000,
    occupations = None,
    nosym       = False,
    conv_thr    = 1e-9,
    mixing_beta = .7,
    wf_collect  = True,
    )

relax_settings = obj(
    sleep         = 3,
    pseudo_dir    = '../pseudos',
    runs          = '',
    results       = '',
    status_only   = 0,
    generate_only = 0,
    account       = '',
    machine       = 'ws32',
    )

def get_relax_job():
    structure = generate_structure(pos_init)
    system = generate_physical_system(structure=generate_structure(pos_init),**valences)
    relax = generate_pwscf(
        job           = job(app='pw.x',cores=4,ppn=4),
        identifier    = 'relax',
        path          = '../relax',
        input_type    = 'relax',
        forc_conv_thr = 1e-4,
        system        = system, 
        **scf_inputs,
    )
    return [relax]
#end def

# settings for the PES sweep
pes_settings = obj(**relax_settings)

def get_pes_job(pos,pstr):
    structure = generate_structure(pos)
    system    = generate_physical_system(structure=generate_structure(pos_init),**valences)
    scf = generate_pwscf(
        job           = job(app='pw.x',cores=4,ppn=4),
        identifier    = 'scf',
        path          = '../scf_pes/scf'+pstr,
        input_type    = 'scf',
        system        = system,
        **scf_inputs,
        )
    return [scf]
#end def


# settings for the main method
main_settings = obj(**srg_settings)

def get_main_job(pos,ls,pstr):
    directory  = '../ls'+str(ls)+'/'+pstr+'/'
    system     = generate_physical_system(structure=generate_structure(pos),C=4)
    qmcpseudos = ['C.ccECP.xml']

    scf = generate_pwscf(
        job        = job(app='pw.x',cores=4,ppn=4),
        path       = directory+'scf',
        identifier = 'scf',
        input_type = 'scf',
        system     = system,
        **scf_inputs,
        )

    p2q = generate_pw2qmcpack(
        identifier   = 'p2q',
        path         = directory+'scf',
        job          = job(app='pw2qmcpack.x',cores=1,ppn=1),
        dependencies = [(scf,'orbitals')],
        )

    system.bconds = 'nnn'
    opt = generate_qmcpack(
        identifier   = 'opt',
        path         = directory+'opt',
        qmc          = 'opt',
        job          = job(app='qmcpack',cores=12,ppn=12),
        system       = system,
        input_type   = 'basic',
        pseudos      = qmcpseudos,
        bconds       = 'nnn',
        J2           = True,
        J1_size      = 10,
        J1_rcut      = 8.0,
        J2_size      = 10,
        J2_rcut      = 8.0,
        minmethod    = 'oneshift',
        blocks       = 1024,
        substeps     = 2,
        steps        = 1,
        cycles       = 20,
        samples      = 256000,
        minwalkers   = 0.05,
        nonlocalpp   = True,
        use_nonlocalpp_deriv = True,
        dependencies = [(p2q,'orbitals') ]
        )

    dmc = generate_qmcpack(
        identifier   = 'dmc',
        path         = directory+'dmc',
        qmc          = 'dmc',
        job          = job(app='qmcpack',cores=16,ppn=16),
        system       = system,
        input_type   = 'basic',
        pseudos      = qmcpseudos,
        bconds       = 'nnn',
        jastrows     = [],
        vmc_samples  = 2000,
        steps        = 200,
        blocks       = 500,
        timestep     = 0.01,
        nonlocalmoves= True,
        ntimesteps   = 1,
        dependencies = [(p2q,'orbitals'),(opt,'jastrow') ]
        )
    return [scf,p2q,opt,dmc]
#end def
