#! /usr/bin/env python3

from nexus import generate_pwscf,generate_pw2qmcpack,generate_qmcpack,job,obj
from coronene_structure import *

E_dim         = 7
valences      = obj(C=4,H=1)
relaxpseudos  = ['C.pbe_v1.2.uspp.F.UPF', 'H.pbe_v1.4.uspp.F.UPF']
qmcpseudos    = ['C.ccECP.xml','H.ccECP.xml'] 
scfpseudos    = ['C.upf','H.upf']
dmc_steps     = 200
dmc_steps_ls  = [1,16]
E_lim_ls      = [0.01,0.001] # Ha
E_lim_pes     = 0.01 # Ry

# jobs for coronene run
# setting for the surrogate job
pseudo_dir = '../pseudos'
nx_account = 'theory'
nx_machine = 'cades'
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
scfjob = obj(app='pw.x',cores=36,ppn=36,presub=presub,hours=2)
optjob = obj(app=qmcapp,cores=36,ppn=36,presub=presub,hours=12)
dmcjob = obj(app=qmcapp,cores=36,ppn=36,presub=presub,hours=48)
p2qjob = obj(app='pw2qmcpack.x',cores=1,ppn=1,presub=presub,minutes=5) 

scf_common = obj(
    input_dft   = 'pbe',
    occupations = None,
    nosym       = False,
    conv_thr    = 1e-9,
    mixing_beta = .7,
    wf_collect  = True,
    )
scf_relax_inputs = obj(
    identifier    = 'relax',
    input_type    = 'relax',
    forc_conv_thr = 1e-4,
    pseudos       = relaxpseudos,
    ecut          = 100,
    ecutrho       = 300,
    **scf_common,
    )
scf_pes_inputs = obj(
    identifier = 'scf',
    input_type = 'scf',
    pseudos    = relaxpseudos,
    ecut       = 100,
    ecutrho    = 300,
    **scf_common,
    )
scf_ls_inputs = obj(
    identifier = 'scf',
    input_type = 'scf',
    pseudos    = scfpseudos,
    ecut       = 300,
    ecutrho    = 2000,
    **scf_common,
    )

relax_settings = obj(
    sleep         = 3,
    pseudo_dir    = pseudo_dir,
    runs          = '',
    results       = '',
    status_only   = 0,
    generate_only = 0,
    account       = nx_account,
    machine       = nx_machine,
    )
def get_relax_job(pos,pstr,cell=cell_init):
    structure = generate_structure(pos,cell)
    system    = generate_physical_system(structure=structure,**valences) 
    relax     = generate_pwscf( 
        system = system,
        job    = job(**scfjob),
        path   = pstr,
        **scf_relax_inputs
        )
    return [relax]
#end def

# settings for the PES sweep
pes_settings = obj(**relax_settings)
def get_pes_job(pos,pstr,cell=cell_init):
    structure = generate_structure(pos,cell)
    system    = generate_physical_system(structure=structure,**valences)
    scf = generate_pwscf(
        system     = system,
        job        = job(**scfjob),
        path       = '../scf_pes/scf'+pstr,
        **scf_pes_inputs,
        )
    return [scf]
#end def

# settings for the line search
opt_inputs = obj(
    identifier   = 'opt',
    qmc          = 'opt',
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
    samples      = 128000,
    minwalkers   = 0.05,
    nonlocalpp   = True,
    use_nonlocalpp_deriv = True,
    )
dmc_inputs = obj(
    identifier   = 'dmc',
    qmc          = 'dmc',
    input_type   = 'basic',
    pseudos      = qmcpseudos,
    bconds       = 'nnn',
    jastrows     = [],
    vmc_samples  = 2000,
    blocks       = 200,
    timestep     = 0.01,
    nonlocalmoves= True,
    ntimesteps   = 1,
    )

ls_settings = obj(**relax_settings)
def get_eqm_jobs(pos,ls,pstr,cell=cell_init):
    directory = '../ls'+str(ls)+'/'+pstr+'/'
    structure = generate_structure(pos,cell)
    system    = generate_physical_system(structure=structure,**valences)

    scf = generate_pwscf(
        system     = system,
        job        = job(**scfjob),
        path       = directory+'scf',
        **scf_ls_inputs,
        )

    p2q = generate_pw2qmcpack(
        identifier   = 'p2q',
        path         = directory+'scf',
        job          = job(**p2qjob),
        dependencies = [(scf,'orbitals')],
        )

    system.bconds = 'nnn'
    opt = generate_qmcpack(
        system       = system,
        path         = directory+'opt',
        job          = job(**optjob),
        dependencies = [(p2q,'orbitals')],
        cycles       = 15,
        **opt_inputs
        )

    dmc = generate_qmcpack(
        system       = system,
        path         = directory+'dmc',
        job          = job(**dmcjob),
        dependencies = [(p2q,'orbitals'),(opt,'jastrow') ],
        **dmc_inputs
        )
    return [scf,p2q,opt,dmc]
#end def

def get_ls_jobs(pos,ls,pstr,eqm_job,cell=cell_init):
    directory = '../ls'+str(ls)+'/'+pstr+'/'
    structure = generate_structure(pos,cell)
    system    = generate_physical_system(structure=structure,**valences)

    scf = generate_pwscf(
        system     = system,
        job        = job(**scfjob),
        path       = directory+'scf',
        **scf_ls_inputs,
        )

    p2q = generate_pw2qmcpack(
        identifier   = 'p2q',
        path         = directory+'scf',
        job          = job(**p2qjob),
        dependencies = [(scf,'orbitals')],
        )

    system.bconds = 'nnn'
    opt = generate_qmcpack(
        system       = system,
        path         = directory+'opt',
        job          = job(**optjob),
        dependencies = [(p2q,'orbitals'),(eqm_job[2],'jastrow')],
        cycles       = 5,
        **opt_inputs
        )

    dmc = generate_qmcpack(
        system       = system,
        path         = directory+'dmc',
        job          = job(**dmcjob),
        dependencies = [(p2q,'orbitals'),(opt,'jastrow') ],
        steps        = dmc_steps*dmc_steps_ls[ls],
        **dmc_inputs
        )
    return [scf,p2q,opt,dmc]
#end def
