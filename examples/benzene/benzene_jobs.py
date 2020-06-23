#! /usr/bin/env python3

from nexus import generate_pwscf,generate_pw2qmcpack,generate_qmcpack,job,obj
from benzene_params import *

E_lim_pes    = 0.01 # Ry
E_lim_ls     = [0.01,0.001] # Ry
dmc_steps_ls = [1,16]
E_dim        = 7

# jobs for benzene run
# setting for the surrogate job
presub = '''
export OMP_NUM_THREADS=1
module purge
module load python/3.6.3
module load PE-intel/3.0
module load intel/18.0.0
module load gcc/6.3.0
module load hdf5_parallel/1.10.3
module load fftw/3.3.5-pe3
module load cmake
module load boost/1.67.0-pe3
module load libxml2/2.9.9
'''
qmcapp = '/home/49t/git/qmcpack/v3.9.2/build_cades_cpu_real_skylake/bin/qmcpack'

scf_inputs = obj(
    input_dft   = 'pbe',
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
    account       = 'qmc',
    machine       = 'cades',
    )

def get_relax_job():
    relax = generate_pwscf(
        job           = job(app='pw.x',cores=36,ppn=36,presub=presub,hours=2),
        identifier    = 'relax',
        path          = '../relax',
        input_type    = 'relax',
        forc_conv_thr = 1e-4,
        ecut          = 100,
        ecutrho       = 300,
        pseudos       = ['C.pbe_v1.2.uspp.F.UPF','H.pbe_v1.4.uspp.F.UPF'], # surrogate with GPRV
        system=generate_physical_system(structure=generate_structure(pos_init),C=4,H=1),
        **scf_inputs,
    )
    return [relax]
#end def

# settings for the PES sweep
pes_settings = obj(**relax_settings)

def get_pes_job(pos,pstr):
    scf = generate_pwscf(
        job           = job(app='pw.x',cores=36,ppn=36,presub=presub,hours=2),
        identifier    = 'scf',
        path          = '../scf_pes/scf'+pstr,
        input_type    = 'scf',
        ecut          = 100,
        ecutrho       = 300,
        pseudos       = ['C.pbe_v1.2.uspp.F.UPF','H.pbe_v1.4.uspp.F.UPF'], # surrogate with GPRV
        system        = generate_physical_system(structure=generate_structure(pos),C=4,H=1),
        **scf_inputs,
        )
    return [scf]
#end def


# settings for the main method
main_settings = obj(**relax_settings)

def get_main_job(pos,ls,pstr):
    directory  = '../ls'+str(ls)+'/'+pstr+'/'
    system     = generate_physical_system(structure=generate_structure(pos),C=4,H=1)
    qmcpseudos = ['C.ccECP.xml','H.ccECP.xml']

    scf = generate_pwscf(
        job        = job(app='pw.x',cores=36,ppn=36,presub=presub,hours=2),
        path       = directory+'scf',
        identifier = 'scf',
        input_type = 'scf',
        system     = system,
        ecut       = 300,
        ecutrho    = 2000,
        pseudos    = ['C.upf','H.upf'], # production with ccECP
        **scf_inputs,
        )

    p2q = generate_pw2qmcpack(
        identifier   = 'p2q',
        path         = directory+'scf',
        job          = job(app='pw2qmcpack.x',cores=1,ppn=1,presub=presub,minutes=5),
        dependencies = [(scf,'orbitals')],
        )

    system.bconds = 'nnn'
    opt = generate_qmcpack(
        identifier   = 'opt',
        path         = directory+'opt',
        qmc          = 'opt',
        job          = job(app=qmcapp,cores=36,ppn=36,presub=presub,hours=12),
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
        blocks       = 512,
        substeps     = 2,
        steps        = 1,
        cycles       = 10,
        samples      = 128000,
        minwalkers   = 0.05,
        nonlocalpp   = True,
        use_nonlocalpp_deriv = True,
        dependencies = [(p2q,'orbitals') ]
        )

    dmc = generate_qmcpack(
        identifier   = 'dmc',
        path         = directory+'dmc',
        qmc          = 'dmc',
        job          = job(app=qmcapp,cores=36,ppn=36,presub=presub,hours=48),
        system       = system,
        input_type   = 'basic',
        pseudos      = qmcpseudos,
        bconds       = 'nnn',
        jastrows     = [],
        vmc_samples  = 2000,
        steps        = 100*dmc_steps_ls[ls],
        blocks       = 200,
        timestep     = 0.01,
        nonlocalmoves= True,
        ntimesteps   = 1,
        dependencies = [(p2q,'orbitals'),(opt,'jastrow') ]
        )
    return [scf,p2q,opt,dmc]
#end def
