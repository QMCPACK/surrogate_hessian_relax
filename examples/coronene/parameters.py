#! /usr/bin/env python3

from numpy import array,diag,linalg,sin,cos,pi,arcsin,arccos
from nexus import generate_pwscf,generate_pw2qmcpack,generate_qmcpack,job,obj,Structure,generate_physical_system

from surrogate_defaults import *
from surrogate_tools import read_geometry

# structure
label       = 'coronene'
cell_init   = array([30.0,30.0,10.0])
pos_xyz = '''
C       17.333479260  16.347254420  5.000000000
C       17.333479260  13.652745580  5.000000000
C       15.000000000  17.694435370  5.000000000
C       19.664315560  17.692928650  5.000000000
C       15.000000000  12.305564630  5.000000000
C       12.666520740  16.347254420  5.000000000
C       19.664315560  12.307071350  5.000000000
C       15.000000000  20.385896360  5.000000000
C       21.961941560  16.299061790  5.000000000
C       19.606008920  20.379716230  5.000000000
C       21.961941560  13.700938210  5.000000000
C       12.666520740  13.652745580  5.000000000
C       17.355987610  21.678675650  5.000000000
C       15.000000000   9.614103640  5.000000000
C       10.335684440  17.692928650  5.000000000
C       19.606008920   9.620283770  5.000000000
C       12.644012390  21.678675650  5.000000000
C       10.335684440  12.307071350  5.000000000
C       17.355987610   8.321324350  5.000000000
C       10.393991080  20.379716230  5.000000000
C       12.644012390   8.321324350  5.000000000
C        8.038058440  16.299061790  5.000000000
C       10.393991080   9.620283770  5.000000000
C        8.038058440  13.700938210  5.000000000
H       23.751175540  17.339410360  5.000000000
H       21.401536810  21.409185650  5.000000000
H       23.751175540  12.660589630  5.000000000
H       17.349782410  23.748412340  5.000000000
H       21.401536810   8.590814350  5.000000000
H       12.650217590  23.748412340  5.000000000
H       17.349782410   6.251587660  5.000000000
H        8.598463190  21.409185650  5.000000000
H       12.650217590   6.251587660  5.000000000
H        6.248824460  17.339410360  5.000000000
H        8.598463190   8.590814350  5.000000000
H        6.248824460  12.660589630  5.000000000
'''
dim           = 3
pos_init,elem = read_geometry(pos_xyz)
masses        = 24*[10947.356792250725] + 12*[918.68110941480279]
num_prt       = len(elem)
jax_hessian   = True

structure_input = obj(
    dim    = 3,
    elem   = elem,
    units  = 'B',
    kgrid  = (1,1,1),
    kshift = (0,0,0),
    )


def pos_to_params(pos,cell=None):
    pos2 = pos.reshape((-1,dim))

    def distance(idx1,idx2):
        return sum((pos2[idx1,:]-pos2[idx2,:])**2)**0.5
    #end def
    p1 = (distance(0,11)+distance(1,5)+distance(2,4))/6
    p2 = (distance(0,3)+distance(1,6)+distance(2,7)+distance(4,13)+distance(5,14)+distance(11,17))/6
    p3 = (distance(3,8)+distance(3,9)+distance(6,10)+distance(6,15)+distance(13,18)+distance(13,20)+distance(17,22)+distance(17,23)+distance(14,21)+distance(14,19)+distance(7,12)+distance(7,16))/12
    p4 = (distance(8,10)+distance(9,12)+distance(15,18)+distance(16,19)+distance(20,22)+distance(21,23))/6
    p5 = (distance(8,24)+distance(9,25)+distance(10,26)+distance(12,27)+distance(15,28)+distance(16,29)+distance(18,30)+distance(19,31)+distance(20,32)+distance(21,33)+distance(22,34)+distance(23,35))/12
    p6 = (distance(24,26)+distance(25,27)+distance(28,30)+distance(29,31)+distance(32,34)+distance(33,35))/6
    params = array([p1,p2,p3,p4,p5,p6])
    return params
#end


# function F
import jax.numpy as jnp
def params_to_pos(p):

    def rotate_xy(angle):
        return jnp.array([[jnp.cos(angle),-jnp.sin(angle),0.0],[jnp.sin(angle),jnp.cos(angle),0.0],[0.0,0.0,1.0]])
    #end def

    y1    = jnp.sin(pi/6)*(p[0]+p[1])-p[3]/2
    alpha = jnp.arccos(y1/p[2])
    x1    = (p[0]+p[1])*jnp.cos(pi/6)+p[2]*jnp.sin(alpha)
    beta  = jnp.arcsin((p[5]-p[3])/2/p[4])
    x2    = x1+p[4]*jnp.cos(beta)

    pos0 = p[0]*jnp.array([jnp.cos(pi/6),jnp.sin(pi/6), 0.0])
    pos1 = (p[0]+p[1])*jnp.array([jnp.cos(pi/6),jnp.sin(pi/6), 0.0])
    pos2 = jnp.array([x1, p[3]/2, 0.0])
    pos3 = jnp.array([x2, p[5]/2, 0.0])
    pos4 = rotate_xy(pi/3) @ jnp.array([x1,-p[3]/2, 0.0])
    pos5 = rotate_xy(pi/3) @ jnp.array([x2,-p[5]/2, 0.0])

    # angles
    a0,a1,a2,a3,a4,a5 = 0*2/3*pi,1/3*pi,2/3*pi,3/3*pi,4/3*pi,5/3*pi

    pos = jnp.array([
        rotate_xy(a0) @ pos0, # C0
        rotate_xy(a5) @ pos0, # C1
        rotate_xy(a1) @ pos0, # C2
        rotate_xy(a0) @ pos1, # C3
        rotate_xy(a4) @ pos0, # C4
        rotate_xy(a2) @ pos0, # C5
        rotate_xy(a5) @ pos1, # C6
        rotate_xy(a1) @ pos1, # C7
        rotate_xy(a0) @ pos2, # C8
        rotate_xy(a0) @ pos4, # C9
        rotate_xy(a5) @ pos4, # C10
        rotate_xy(a3) @ pos0, # C11
        rotate_xy(a1) @ pos2, # C12
        rotate_xy(a4) @ pos1, # C13
        rotate_xy(a2) @ pos1, # C14
        rotate_xy(a5) @ pos2, # C15
        rotate_xy(a1) @ pos4, # C16
        rotate_xy(a3) @ pos1, # C17
        rotate_xy(a4) @ pos4, # C18
        rotate_xy(a2) @ pos2, # C19
        rotate_xy(a4) @ pos2, # C20
        rotate_xy(a2) @ pos4, # C21
        rotate_xy(a3) @ pos4, # C22
        rotate_xy(a3) @ pos2, # C23
        rotate_xy(a0) @ pos3, # H01/24
        rotate_xy(a0) @ pos5, # H02/25
        rotate_xy(a5) @ pos5, # H03/26
        rotate_xy(a1) @ pos3, # H04/27
        rotate_xy(a5) @ pos3, # H05/28
        rotate_xy(a1) @ pos5, # H06/29
        rotate_xy(a4) @ pos5, # H07/30
        rotate_xy(a2) @ pos3, # H08/31
        rotate_xy(a4) @ pos3, # H09/32
        rotate_xy(a2) @ pos5, # H10/33
        rotate_xy(a3) @ pos5, # H11/34
        rotate_xy(a3) @ pos3, # H12/35
    ]).reshape(-1)

    pos = (pos.reshape(-1,3)+cell_init/2).reshape(-1)

    return pos
#end def

# jobs
valences       = obj(C=4,H=1)
relaxpseudos   = ['C.pbe_v1.2.uspp.F.UPF', 'H.pbe_v1.4.uspp.F.UPF']
qmcpseudos     = ['C.ccECP.xml','H.ccECP.xml']
scfpseudos     = ['C.upf','H.upf']
steps_times_error2 = 0.0003 # (steps-1)* error**2

# setting for the surrogate job
pseudo_dir = '../pseudos'
nx_account = 'qmc'
nx_machine = 'cades'
cores      = 36
presub = '''
export OMP_NUM_THREADS=1
module purge
module load python/3.6.3
module load PE-intel/3.0
module load intel/18.0.0
module load gcc/6.3.0
module load hdf5_parallel/1.8.17
module load fftw/3.3.5
module load cmake
module load boost/1.67.0
module load libxml2/2.9.9
'''
qmcapp = '/home/49t/git/qmcpack/latest/build_cades_cpu_real_skylake/bin/qmcpack'
scfjob = obj(app='pw.x',cores=cores,ppn=cores,presub=presub,hours=2,queue='burst')
optjob = obj(app=qmcapp,cores=cores,ppn=cores,presub=presub,hours=12)
dmcjob = obj(app=qmcapp,cores=cores,ppn=cores,presub=presub,hours=12)
p2qjob = obj(app='pw2qmcpack.x',cores=1,ppn=1,presub=presub,minutes=5,queue='burst')

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
    disk_io       = 'none',
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
    use_nonlocalpp_deriv = False, # too heavy
    meshfactor   = 0.8,
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
    meshfactor   = 0.8,
    warmupsteps  = 100,
    )

# nexus settings
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

# SCF relax job
def get_relax_job(pos,pstr,**kwargs):
    relax = generate_pwscf(
        system = get_system(pos),
        job    = job(**scfjob),
        path   = pstr,
        **scf_relax_inputs
        )
    return [relax]
#end def

# SCF line search
def get_scf_pes_job(pos,path,**kwargs):
    scf = generate_pwscf(
        system  = get_system(pos),
        job     = job(**scfjob),
        path    = path,
        **scf_pes_inputs,
        )
    return [scf]
#end def

# DMC line search
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
