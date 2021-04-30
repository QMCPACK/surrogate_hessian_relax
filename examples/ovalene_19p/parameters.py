#! /usr/bin/env python3

# Ovalene (19 parameters)
#
# This is an example of configuring the structural mappings and computing jobs based on the original publication
# The file is not (yet) fully curated for pedagogical purposes, and may not reflect the latest good practices of composing the parameter file.
# However, it serves to demonstrate that the implementation in parameters.py can be done in any almost any style, as long as it defines
#   Starting structure (pos_init as 1D array)
#   Consistent parameter mappings: (pos_to_params, params_to_pos)
#   Functions returning Nexus workflows (get_relax_job,get_scf_pes_job,,get_dmc_jobs)

from numpy import array,diag,linalg,mean
from scipy.optimize import minimize
from nexus import generate_pwscf,generate_pw2qmcpack,generate_qmcpack,job,obj,Structure,generate_physical_system

from surrogate_tools import read_geometry
from surrogate_defaults import *

# structure
label       = 'coronene'
cell_init   = array([30.0,34.0,10.0])
pos_xyz = '''
C       -1.41140       -1.21947     0.00000
C       -0.70554        0.00000     0.00000
C       -0.70549       -2.44067     0.00000
C       -2.82376       -1.21967     0.00000
C        0.70554       -0.00000     0.00000
C        0.70549       -2.44067     0.00000
C       -1.41140        1.21947     0.00000
C       -1.41073       -3.66292     0.00000
C       -3.52245        0.00000     0.00000
C       -3.51563       -2.44159     0.00000
C       -2.82376        1.21967     0.00000
C        1.41140       -1.21947     0.00000
C       -2.81459       -3.65377     0.00000
C        1.41140        1.21947     0.00000
C        1.41073       -3.66292     0.00000
C       -0.70549        2.44067     0.00000
C       -0.70016       -4.87383     0.00000
C        2.82376       -1.21967     0.00000
C        0.70549        2.44067     0.00000
C        0.70016       -4.87383     0.00000
C       -3.51563        2.44159     0.00000
C        2.82376        1.21967     0.00000
C        2.81459       -3.65377     0.00000
C       -1.41073        3.66292     0.00000
C        3.52245       -0.00000     0.00000
C        3.51563       -2.44159     0.00000
C       -2.81459        3.65377     0.00000
C        1.41073        3.66292     0.00000
C        3.51563        2.44159     0.00000
C       -0.70016        4.87383     0.00000
C        2.81459        3.65377     0.00000
C        0.70016        4.87383     0.00000
H       -3.36826        4.58512     0.00000
H       -1.22930        5.81934     0.00000
H        3.36826        4.58512     0.00000
H        1.22930        5.81934     0.00000
H       -4.60714        0.00000     0.00000
H       -4.59902       -2.45747     0.00000
H       -3.36826       -4.58512     0.00000
H       -1.22930       -5.81934     0.00000
H        1.22930       -5.81934     0.00000
H       -4.59902        2.45747     0.00000
H        3.36826       -4.58512     0.00000
H        4.60714       -0.00000     0.00000
H        4.59902       -2.45747     0.00000
H        4.59902        2.45747     0.00000
'''
dim           = 3
pos_init,elem = read_geometry(pos_xyz)
pos_init      = ((pos_init/0.529189379).reshape(-1,3) + cell_init/2).reshape(-1) # to bohr and add cell
masses        = 32*[10947.356792250725] + 14*[918.68110941480279]
relax_cell    = False
num_prt       = len(elem)

structure_input = obj(
    dim    = 3,
    elem   = elem,
    units  = 'B',
    kgrid  = (1,1,1),
    kshift = (0,0,0),
    )

def pos_to_params(pos,**kwargs):
    pos2 = pos.reshape((-1,dim))

    def distance(idx1,idx2):
        return sum((pos2[idx1,:]-pos2[idx2,:])**2)**0.5
    #end def

    r0  = distance(1,4)
    r1  = mean([distance(15,18),distance(2,5)])
    r2  = mean([distance(29,31),distance(16,19)])
    r3  = mean([distance(4,13) ,distance(1,6)  ,distance(0,1)  ,distance(4,11) ])
    r4  = mean([distance(13,18),distance(6,15) ,distance(0,2)  ,distance(5,11) ])
    r5  = mean([distance(15,23),distance(18,27),distance(5,14) ,distance(2,7)  ])
    r6  = mean([distance(23,29),distance(27,31),distance(7,16) ,distance(14,19)])
    r7  = mean([distance(13,21),distance(10,6) ,distance(3,0)  ,distance(10,6) ])
    r8  = mean([distance(26,23),distance(27,30),distance(14,22),distance(12,7) ])
    r9  = mean([distance(21,24),distance(24,17),distance(8,10) ,distance(8,3)  ])
    r10 = mean([distance(21,28),distance(25,17),distance(20,10),distance(9,3)  ])
    r11 = mean([distance(30,28),distance(25,22),distance(20,26),distance(9,12) ])
    r12 = mean([distance(24,43),distance( 8,36)])
    r13 = mean([distance(28,45),distance(20,41),distance( 9,37),distance(25,44)])
    r14 = mean([distance(30,34),distance(26,32),distance(12,38),distance(22,42)])
    r15 = mean([distance(31,35),distance(29,33),distance(19,40),distance(16,39)])
    r16 = mean([distance(43,45),distance(43,44),distance(36,37),distance(36,41)])
    r17 = mean([distance(34,45),distance(32,41),distance(37,38),distance(42,44)])
    r18 = mean([distance(34,35),distance(32,33),distance(38,39),distance(40,42)])
    params = array([r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18])
    return params
#end 

# function F
def params_to_pos(p, return_opt=False, **kwargs):
    
    q0 = array([[ 1.0,0.0],[0.0, 1.0],[0.0,0.0]]).T
    q1 = array([[-1.0,0.0],[0.0, 1.0],[0.0,0.0]]).T
    q2 = array([[-1.0,0.0],[0.0,-1.0],[0.0,0.0]]).T
    q3 = array([[ 1.0,0.0],[0.0,-1.0],[0.0,0.0]]).T

    pos_init2 = pos_init.reshape(-1,3)

    # auxiliary position variable, constrain to xy plane
    aux_pos0 = array([
        #pos_init2[4 ,:2]-cell_init[:2]/2, #0 # contrain pos0
        pos_init2[13,0]-cell_init[0]/2, #1x  // 1
        pos_init2[13,1]-cell_init[1]/2, #1y  
        pos_init2[18,0]-cell_init[0]/2, #2x  // 2
        pos_init2[18,1]-cell_init[1]/2, #2y
        pos_init2[27,0]-cell_init[0]/2, #3x  // 3
        pos_init2[27,1]-cell_init[1]/2, #3y
        pos_init2[31,0]-cell_init[0]/2, #4x  // 4
        pos_init2[31,1]-cell_init[1]/2, #4y
        pos_init2[21,0]-cell_init[0]/2, #6x  // 5
        pos_init2[21,1]-cell_init[1]/2, #6y
        pos_init2[28,0]-cell_init[0]/2, #7x  // 6
        pos_init2[28,1]-cell_init[1]/2, #7y
        pos_init2[30,0]-cell_init[0]/2, #8x  // 7
        pos_init2[30,1]-cell_init[1]/2, #8y
        pos_init2[45,0]-cell_init[0]/2, #10x  // 8
        pos_init2[45,1]-cell_init[1]/2, #10y
        pos_init2[34,0]-cell_init[0]/2, #11x  // 9
        pos_init2[34,1]-cell_init[1]/2, #11y
        pos_init2[35,0]-cell_init[0]/2, #12x  // 10
        pos_init2[35,1]-cell_init[1]/2, #12y
        pos_init2[24, 0]-cell_init[ 0]/2, #5 # constrain pos y
        pos_init2[43, 0]-cell_init[ 0]/2, #9x # constraint pos y
        ])

    # xy coordinates only
    def aux_pos_to_par_diff(pos):
        posq2 = pos[0:20].reshape(-1,2)
        def distance(idx1,idx2):
            return sum((posq2[idx1-1,:]-posq2[idx2-1,:])**2)**0.5
        #end def
        pq = []
        pq.append( p[0] )            #0
        pq.append( 2*posq2[1,0] )    #1
        pq.append( 2*posq2[3,0] )    #2
        pq.append( (posq2[0,1]**2+(posq2[0,0]-p[0]/2)**2)**0.5 )   #3
        pq.append( distance(1,2) )   #4
        pq.append( distance(2,3) )   #5
        pq.append( distance(3,4) )   #6
        pq.append( distance(1,5) )   #7
        pq.append( distance(3,7) )   #8
        pq.append( (posq2[4,1]**2+(posq2[4,0]-pos[20])**2)**0.5 )   #9
        pq.append( distance(5,6) )   #10
        pq.append( distance(6,7) )   #11
        pq.append( pos[21]-pos[20] ) #12
        pq.append( distance(6,8) )   #13
        pq.append( distance(7,9))    #14
        pq.append( distance(4,10))   #15
        pq.append( (posq2[7,1]**2+(pos[21]-posq2[7,0])**2)**0.5 )   #16
        pq.append( distance(8,9))    #17
        pq.append( distance(9,10))   #18
        pq = array(pq)
        err = sum((pq-p)**2)
        return err
    #end def

    aux_pos_res = minimize(aux_pos_to_par_diff,aux_pos0,tol=1e-7,method='BFGS') # Powell used to work, BFGS
    aux_pos = array(aux_pos_res.x)

    posq = array([
       [p[0]/2, 0.0]      , # 0
        aux_pos[0:2]      , # 1
        aux_pos[2:4]      , # 2
        aux_pos[4:6]      , # 3
        aux_pos[6:8]      , # 4
       [aux_pos[20],0.0]  , # 5
        aux_pos[8:10]     , # 6
        aux_pos[10:12]    , # 7
        aux_pos[12:14]    , # 8
       [aux_pos[21],0.0]  , # 9
        aux_pos[14:16]    , # 10
        aux_pos[16:18]    , # 11
        aux_pos[18:20]    , # 12
        ])

    # z dimension is added in multiplication
    pos = array([
        posq[1]  @ q2 +cell_init/2, # C0
        posq[0]  @ q2 +cell_init/2, # C1
        posq[2]  @ q2 +cell_init/2, # C2
        posq[6]  @ q2 +cell_init/2, # C3
        posq[0]  @ q0 +cell_init/2, # C4
        posq[2]  @ q3 +cell_init/2, # C5
        posq[1]  @ q1 +cell_init/2, # C6
        posq[3]  @ q2 +cell_init/2, # C7
        posq[5]  @ q2 +cell_init/2, # C8
        posq[7]  @ q2 +cell_init/2, # C9
        posq[6]  @ q1 +cell_init/2, # C10
        posq[1]  @ q3 +cell_init/2, # C11
        posq[8]  @ q2 +cell_init/2, # C12
        posq[1]  @ q0 +cell_init/2, # C13
        posq[3]  @ q3 +cell_init/2, # C14
        posq[2]  @ q1 +cell_init/2, # C15
        posq[4]  @ q2 +cell_init/2, # C16
        posq[6]  @ q3 +cell_init/2, # C17
        posq[2]  @ q0 +cell_init/2, # C18
        posq[4]  @ q3 +cell_init/2, # C19
        posq[7]  @ q1 +cell_init/2, # C20
        posq[6]  @ q0 +cell_init/2, # C21
        posq[8]  @ q3 +cell_init/2, # C22
        posq[3]  @ q1 +cell_init/2, # C23
        posq[5]  @ q0 +cell_init/2, # C24
        posq[7]  @ q3 +cell_init/2, # C25
        posq[8]  @ q1 +cell_init/2, # C26
        posq[3]  @ q0 +cell_init/2, # C27
        posq[7]  @ q0 +cell_init/2, # C28
        posq[4]  @ q1 +cell_init/2, # C29
        posq[8]  @ q0 +cell_init/2, # C30
        posq[4]  @ q0 +cell_init/2, # C31
        posq[11] @ q1 +cell_init/2, # H1 //32 
        posq[12] @ q1 +cell_init/2, # H1 //33 
        posq[11] @ q0 +cell_init/2, # H1 //34 
        posq[12] @ q0 +cell_init/2, # H1 //35 
        posq[9]  @ q2 +cell_init/2, # H1 //36 
        posq[10] @ q2 +cell_init/2, # H1 //37 
        posq[11] @ q2 +cell_init/2, # H1 //38 
        posq[12] @ q2 +cell_init/2, # H1 //39 
        posq[12] @ q3 +cell_init/2, # H1 //40 
        posq[10] @ q1 +cell_init/2, # H1 //41 
        posq[11] @ q3 +cell_init/2, # H1 //42 
        posq[9]  @ q0 +cell_init/2, # H1 //43 
        posq[10] @ q3 +cell_init/2, # H1 //44 
        posq[10] @ q0 +cell_init/2, # H1 //45 
    ]).reshape(-1)

    if return_opt:
        return pos,aux_pos_res
    else:
        return pos
    #end if
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
dmcjob4    = obj(app=qmcapp,cores=4*cores,nodes=4,ppn=cores,presub=presub,hours=12)
p2qjob     = obj(app=p2qapp,cores=1,ppn=1,presub=presub,minutes=5)

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
steps_times_error2 = 0.0003 # (steps-1)* error**2
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

    if dmcsteps < 1000:
        dmc = generate_qmcpack(
            system       = system,
            path         = path+'/dmc',
            job          = job(**dmcjob),
            dependencies = [(p2q,'orbitals'),(opt,'jastrow') ],
            steps        = dmcsteps,
            **dmc_inputs
            )
        return [scf,p2q,opt,dmc]
    else: # 4 nodes
        dmc = generate_qmcpack(
            system       = system,
            path         = path+'/dmc',
            job          = job(**dmcjob4),
            dependencies = [(p2q,'orbitals'),(opt,'jastrow') ],
            steps        = dmcsteps,
            **dmc_inputs
            )
        return [scf,p2q,opt,dmc]
    #end if
#end def
