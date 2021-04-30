#! /usr/bin/env python3

# Benzene (2 parameters; CC/CH)
# 
# This is an example of configuring the structural mappings and computing jobs based on the original publication
# The file is not (yet) fully curated for pedagogical purposes, and may not reflect the latest good practices of composing the parameter file.
# However, it serves to demonstrate that the implementation in parameters.py can be done in any almost any style, as long as it defines
#   Starting structure (pos_init as 1D array)
#   Consistent parameter mappings: (pos_to_params, params_to_pos)
#   Functions returning Nexus workflows (get_relax_job,get_scf_pes_job,get_scf_ls_job,get_dmc_jobs)

from numpy import array,sin,cos,pi,diag,linalg
from nexus import generate_pwscf,generate_pw2qmcpack,generate_qmcpack,job,obj,Structure,generate_physical_system

from surrogate_defaults import *
from surrogate_tools import read_geometry

# Structure
label      = 'benzene'
cell_init  = array([20.0,20.0,10.0])
pos_xyz    = '''
C  0.          2.65075664  0.
H  0.          4.70596609  0.
C -2.29562041  1.32537832  0.
H -4.07549676  2.35299249  0.
C -2.29562041 -1.32537832  0.
H -4.07549676 -2.3529925   0.
C  0.         -2.65075664  0.
H  0.         -4.70596609  0.
C  2.29562041 -1.32537832  0.
H  4.07549676 -2.3529925   0.
C  2.29562041  1.32537832  0.
H  4.07549676  2.35299249  0.
'''
dim           = 3
pos_init,elem = read_geometry(pos_xyz)
pos_init      = (pos_init.reshape(-1,3) + cell_init/2).reshape(-1) # add cell
masses        = 6*[10947.356792250725,918.68110941480279]
relax_cell    = False
num_prt       = len(elem)

structure_input = obj(
    dim    = 3,
    elem   = elem,
    units  = 'B',
    kgrid  = (1,1,1),
    kshift = (0,0,0),
    )

# param 1: CC distance (no shift H)
r_CC = array([cos( 3*pi/6), sin( 3*pi/6), 0., 0., 0., 0.,
              cos( 5*pi/6), sin( 5*pi/6), 0., 0., 0., 0.,
              cos( 7*pi/6), sin( 7*pi/6), 0., 0., 0., 0.,
              cos( 9*pi/6), sin( 9*pi/6), 0., 0., 0., 0.,
              cos(11*pi/6), sin(11*pi/6), 0., 0., 0., 0.,
              cos(13*pi/6), sin(13*pi/6), 0., 0., 0., 0.,])/6
# param 2 : CH distance
r_CH = array([-cos( 3*pi/6), -sin( 3*pi/6), 0.,cos( 3*pi/6), sin(3*pi/6), 0.,
              -cos( 5*pi/6), -sin( 5*pi/6), 0.,cos( 5*pi/6), sin(5*pi/6), 0.,
              -cos( 7*pi/6), -sin( 7*pi/6), 0.,cos( 7*pi/6), sin(7*pi/6), 0.,
              -cos( 9*pi/6), -sin( 9*pi/6), 0.,cos( 9*pi/6), sin(9*pi/6), 0.,
              -cos(11*pi/6), -sin(11*pi/6), 0.,cos(11*pi/6),sin(11*pi/6), 0.,
              -cos(13*pi/6), -sin(13*pi/6), 0.,cos(13*pi/6),sin(13*pi/6), 0.,])/6
r_HH = array([0., 0., 0.,cos( 3*pi/6), sin(3*pi/6), 0.,
              0., 0., 0.,cos( 5*pi/6), sin(5*pi/6), 0.,
              0., 0., 0.,cos( 7*pi/6), sin(7*pi/6), 0.,
              0., 0., 0.,cos( 9*pi/6), sin(9*pi/6), 0.,
              0., 0., 0.,cos(11*pi/6),sin(11*pi/6), 0.,
              0., 0., 0.,cos(13*pi/6),sin(13*pi/6), 0.,])/6
#delta_pinv = array([r_CC,r_HH])
delta_pinv = array([r_CC,r_CH])
delta      = linalg.pinv(delta_pinv)

def pos_to_params(pos):
    params = delta_pinv @ pos
    return params
#end def

def params_to_pos(params):
    #pos = delta @ params
    pos = ((delta @ params).reshape(-1,3) + cell_init/2).flatten()
    return pos
#end def

# Pseudos
valences       = obj(C=4,H=1)
relaxpseudos   = ['C.pbe_v1.2.uspp.F.UPF', 'H.pbe_v1.4.uspp.F.UPF']
qmcpseudos     = ['C.ccECP.xml','H.ccECP.xml']
scfpseudos     = ['C.upf','H.upf']

# Setting for the nexus jobs based on file layout and computing environment
pseudo_dir = '../pseudos'
nx_machine = 'ws8'
cores      = 8
presub     = ''
qmcapp     = '/path/to/qmcpack'
qeapp      = '/path/to/pw.x'
p2qapp     = '/path/to/pw2qmcpack.x'
scfjob     = obj(app=qeapp, cores=cores,ppn=cores,presub=presub,hours=2)
optjob     = obj(app=qmcapp,cores=cores,ppn=cores,presub=presub,hours=12)
dmcjob     = obj(app=qmcapp,cores=cores,ppn=cores,presub=presub,hours=12)
p2qjob     = obj(app=p2qapp,cores=1,ppn=1,presub=presub,minutes=5)

scf_common = obj(
    input_dft   = 'pbe',
    occupations = None,
    nosym       = False,
    conv_thr    = 1e-9,
    mixing_beta = .7,
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
    disk_io    = 'none',
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
opt_inputs = obj(
    identifier   = 'opt',
    qmc          = 'opt',
    input_type   = 'basic',
    pseudos      = qmcpseudos,
    bconds       = 'nnn',
    J2           = True,
    J1_size      = 6,
    J1_rcut      = 6.0,
    J2_size      = 6,
    J2_rcut      = 6.0,
    minmethod    = 'oneshift',
    blocks       = 512,
    substeps     = 2,
    steps        = 1,
    samples      = 128000,
    minwalkers   = 0.05,
    nonlocalpp   = True,
    use_nonlocalpp_deriv = False,
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
nx_settings = obj(
    sleep         = 3,
    pseudo_dir    = pseudo_dir,
    runs          = '',
    results       = '',
    status_only   = 0,
    generate_only = 0,
    account       = nx_account,
    machine       = nx_machine,
    )


# construct system based on position
def get_system(pos,cell=cell_init):
    structure = Structure(
        pos    = pos.reshape((-1,dim)),
        axes   = diag(cell),
        **structure_input,
        )
    return generate_physical_system(structure=structure,**valences)
#end def

def get_relax_job(pos,pstr,**kwargs):
    relax     = generate_pwscf(
        system = get_system(pos),
        job    = job(**scfjob),
        path   = pstr,
        **scf_relax_inputs
        )
    return [relax]
#end def

# settings for the PES sweep
def get_scf_pes_job(pos,path,**kwargs):
    scf = generate_pwscf(
        system     = get_system(pos),
        job        = job(**scfjob),
        path       = path,
        **scf_pes_inputs,
        )
    return [scf]
#end def

# settings for the SCF ls
def get_scf_ls_job(pos,path,**kwargs):
    scf = generate_pwscf(
        system     = get_system(pos),
        job        = job(**scfjob),
        path       = path,
        **scf_ls_inputs,
        )
    return [scf]
#end def

# DMC line search
steps_times_error2 = 4.8e-05 # (steps-1)* error**2
def get_dmc_jobs(pos,path,sigma,jastrow=None,**kwargs):
    system   = get_system(pos)
    dmcsteps = int(steps_times_error2/sigma**2)+1

    scf = generate_pwscf(
        system     = system,
        job        = job(**scfjob),
        path       = path+'/scf',
        **scf_ls_inputs,
        )

    p2q = generate_pw2qmcpack(
        identifier   = 'p2q',
        path         = path+'/scf',
        job          = job(**p2qjob),
        dependencies = [(scf,'orbitals')],
        )

    system.bconds = 'nnn'
    if jastrow is None:
        opt_cycles = 16
    else:
        opt_cycles = 8
    #end if
    opt = generate_qmcpack(
        system       = system,
        path         = path+'/opt',
        job          = job(**optjob),
        dependencies = [(p2q,'orbitals')],
        cycles       = opt_cycles,
        **opt_inputs
        )
    if not jastrow is None:
        opt.depends((jastrow,'jastrow'))
    #end if

    dmc = generate_qmcpack(
        system       = system,
        path         = path+'/dmc',
        job          = job(**dmcjob),
        dependencies = [(p2q,'orbitals'),(opt,'jastrow') ],
        steps        = dmcsteps,
        **dmc_inputs
        )
    return [scf,p2q,opt,dmc]
#end def
