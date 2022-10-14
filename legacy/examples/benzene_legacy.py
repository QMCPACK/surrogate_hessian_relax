#!/usr/bin/env python3

# Benzene: line-search example
#   2-parameter problem: CC/CH bond lengths
#   Surrogate theory: DFT (PBE)
#   Stochastic tehory: DMC
#
# This example executes a standard line-search minimization workflow for the 
# ground state of the benzene molecule, using DFT (PBE) as the surrogate
# method and Diffusion Monte Carlo (DMC) as the stochastic method.
#
# Computing task: Suitable for institutional clusters

# First, the user must set up Nexus according to their computing environment.

from nexus import generate_pwscf, generate_qmcpack, job, obj, Structure, run_project
from nexus import generate_pw2qmcpack, generate_physical_system, settings

# Modify the below variables as needed
cores      = 24
presub     = ''
qeapp      = '/path/to/pw.x'
p2qapp     = '/path/to/pw2qmcpack.x'
qmcapp     = '/path/to/qmcpack'
scfjob     = obj(app=qeapp, cores=cores,ppn=cores,presub=presub,hours=2)
p2qjob     = obj(app=p2qapp,cores=1,ppn=1,presub=presub,minutes=5)
optjob     = obj(app=qmcapp,cores=cores,ppn=cores,presub=presub,hours=12)
dmcjob     = obj(app=qmcapp,cores=cores,ppn=cores,presub=presub,hours=12)
nx_settings = obj(
    sleep         = 3,
    pseudo_dir    = 'pseudos',
    runs          = '',
    results       = '',
    status_only   = 0,
    generate_only = 0,
    machine       = 'ws24',
    )
settings(**nx_settings) # initiate nexus

# Pseudos (execute download_pseudos.sh in the working directory)
relaxpseudos   = ['C.pbe_v1.2.uspp.F.upf', 'H.pbe_v1.4.uspp.F.upf']
qmcpseudos     = ['C.ccECP.xml','H.ccECP.xml']
scfpseudos     = ['C.upf','H.upf']


# Implement the following parametric mappings for benzene
#   p0: C-C distance
#   p1: C-H distance

from numpy import mean,array,sin,pi,cos,diag,linalg

# Forward mapping: produce parameter values from an array of atomic positions
def pos_to_params(pos):
    pos = pos.reshape(-1,3) # make sure of the shape
    # for easier comprehension, list particular atoms
    C0 = pos[0]
    C1 = pos[1]
    C2 = pos[2]
    C3 = pos[3]
    C4 = pos[4]
    C5 = pos[5]
    H0 = pos[6]
    H1 = pos[7]
    H2 = pos[8]
    H3 = pos[9]
    H4 = pos[10]
    H5 = pos[11]
    def distance(r1,r2):
        return sum((r1-r2)**2)**0.5
    #end def
    # for redundancy, calculate mean bond lengths
    # 0) from neighboring C-atoms
    r_CC = mean([distance(C0,C1),
                 distance(C1,C2),
                 distance(C2,C3),
                 distance(C3,C4),
                 distance(C4,C5),
                 distance(C5,C0)])
    # 1) from corresponding H-atoms
    r_CH = mean([distance(C0,H0),
                 distance(C1,H1),
                 distance(C2,H2),
                 distance(C3,H3),
                 distance(C4,H4),
                 distance(C5,H5)])
    params = array([r_CC,r_CH])
    return params
#end def

# Backward mapping: produce array of atomic positions from parameters
axes = array([20,20,10]) # simulate in vacuum
def params_to_pos(params):
    r_CC = params[0]
    r_CH = params[1]
    # place atoms on a hexagon in the xy-directions
    hex_xy = array([[cos( 3*pi/6), sin( 3*pi/6), 0.],
                    [cos( 5*pi/6), sin( 5*pi/6), 0.],
                    [cos( 7*pi/6), sin( 7*pi/6), 0.],
                    [cos( 9*pi/6), sin( 9*pi/6), 0.],
                    [cos(11*pi/6), sin(11*pi/6), 0.],
                    [cos(13*pi/6), sin(13*pi/6), 0.]])
    pos_C = axes/2+hex_xy*r_CC # C-atoms are one C-C length apart from origin
    pos_H = axes/2+hex_xy*(r_CC+r_CH) # H-atoms one C-H length apart from C-atoms
    pos = array([pos_C,pos_H]).flatten()
    return pos
#end def

# Guess initial parameter values
p_init = array([2.651,2.055])
pos_init = params_to_pos(p_init)
elem = 6*['C']+6*['H']

# Check consistency of the mappings
from surrogate_tools import check_mapping_consistency
if check_mapping_consistency(p_init,pos_to_params,params_to_pos):
    print('Parameter mappings are consistent at the equilibrium!')
#end if


# The following data structures provide simulation inputs for each theory

# relaxation job
scf_relax_inputs = obj(
    input_dft     = 'pbe',
    occupations   = None,
    nosym         = False,
    conv_thr      = 1e-9,
    mixing_beta   = .7,
    identifier    = 'relax',
    input_type    = 'relax',
    forc_conv_thr = 1e-4,
    pseudos       = relaxpseudos,
    ecut          = 100,
    ecutrho       = 300,
    )
# single-shot energy on the same PES as relax
scf_pes_inputs = obj(
    input_dft   = 'pbe',
    occupations = None,
    nosym       = False,
    conv_thr    = 1e-9,
    mixing_beta = .7,
    identifier  = 'scf',
    input_type  = 'scf',
    pseudos     = relaxpseudos,
    ecut        = 100,
    ecutrho     = 300,
    disk_io     = 'none',
    )
# SCF settings for QMC: different pseudopotential and higher ecut
scf_qmc_inputs = obj(
    input_dft   = 'pbe',
    occupations = None,
    nosym       = False,
    conv_thr    = 1e-9,
    mixing_beta = .7,
    identifier  = 'scf',
    input_type  = 'scf',
    pseudos     = scfpseudos,
    ecut        = 300,
    ecutrho     = 600,
    )
# Inputs for QMC optimizer
opt_inputs = obj(
    identifier   = 'opt',
    qmc          = 'opt',
    input_type   = 'basic',
    pseudos      = qmcpseudos,
    bconds       = 'nnn',
    J2           = True,
    J1_size      = 6,
    J1_rcut      = 6.0,
    J2_size      = 8,
    J2_rcut      = 8.0,
    minmethod    = 'oneshift',
    blocks       = 200,
    substeps     = 2,
    steps        = 1,
    samples      = 100000,
    minwalkers   = 0.1,
    nonlocalpp   = True,
    use_nonlocalpp_deriv = False,
    )
# Inputs for DMC
dmc_inputs = obj(
    identifier   = 'dmc',
    qmc          = 'dmc',
    input_type   = 'basic',
    pseudos      = qmcpseudos,
    bconds       = 'nnn',
    jastrows     = [],
    vmc_samples  = 1000,
    blocks       = 200,
    timestep     = 0.01,
    nonlocalmoves= True,
    ntimesteps   = 1,
    )

# construct Nexus system based on position

def get_system(pos):
    structure = Structure(
        pos    = pos.reshape((-1,3)),
        axes   = diag(axes),
        dim    = 3,
        elem   = elem,
        units  = 'B',
        kgrid  = (1,1,1),
        kshift = (0,0,0),
        )
    return generate_physical_system(structure=structure,C=4,H=1)
#end def

# return a 1-item list of Nexus jobs: SCF relaxation
def get_relax_job(pos,path,**kwargs):
    relax     = generate_pwscf(
        system = get_system(pos),
        job    = job(**scfjob),
        path   = path,
        **scf_relax_inputs
        )
    return [relax]
#end def

# return a 3-item list of Nexus jobs: SCF phonon calculation
# Since the phonon calculations are not standard in Nexus, we are providing the 
# inputs manually by using GenericSimulation and input_template classes
from simulation import GenericSimulation,input_template
phjob    = obj(app_command='ph.x -in phonon.in', cores=cores,ppn=cores,presub=presub,hours=2)
q2rjob   = obj(app_command='q2r.x -in q2r.in', cores=cores,ppn=cores,presub=presub,hours=2)
ph_input = input_template('''
phonons at gamma
&inputph
   outdir          = 'pwscf_output'
   prefix          = 'pwscf'
   fildyn          = 'PH.dynG'
   ldisp           = .true.
   tr2_ph          = 1.0d-12,
   nq1             = 1
   nq2             = 1
   nq3             = 1
/
''')
q2r_input = input_template('''
&input
  fildyn='PH.dynG', zasr='zero-dim', flfrc='FC.fc'
/
''')
def get_phonon_jobs(pos,path,**kwargs):
    scf = generate_pwscf(
        system     = get_system(pos),
        job        = job(**scfjob),
        path       = path,
        **scf_pes_inputs,
        )
    scf.input.control.disk_io = 'low' # write charge density
    phonon = GenericSimulation(
        system       = get_system(pos),
        job          = job(**phjob),
        path         = path,
        input        = ph_input,
        identifier   = 'phonon',
        dependencies = (scf,'other'),
        )
    q2r = GenericSimulation(
        system     = get_system(pos),
        job        = job(**q2rjob),
        path       = path,
        input      = q2r_input,
        identifier = 'q2r',
        dependencies = (phonon,'other'),
        )
    return [scf,phonon,q2r]
#end def

# return a 1-item list of Nexus jobs: single-point PES
def get_scf_pes_job(pos,path,**kwargs):
    scf = generate_pwscf(
        system     = get_system(pos),
        job        = job(**scfjob),
        path       = path,
        **scf_pes_inputs,
        )
    return [scf]
#end def

# return a 1-item list of Nexus jobs: single-point SCF for QMC
def get_scf_qmc_job(pos,path,**kwargs):
    scf = generate_pwscf(
        system     = get_system(pos),
        job        = job(**scfjob),
        path       = path,
        **scf_qmc_inputs,
        )
    return [scf]
#end def

# Return a 4-item list of Nexus jobs:
#  1) scf: orbitals
#  2) p2q: conversion for QMCPACK
#  3) opt: Jastrow optimization
#  4) dmc: DMC calculation
#
# sigma parameter needed for target accuracy
# if jastrow job is provided, use as a starting point
steps_times_error2 = 1.2e-04 # (steps-1)* error**2
def get_dmc_jobs(pos,path,sigma,jastrow=None,**kwargs):
    system   = get_system(pos)
    # Estimate the number of steps necessary to produce the desired accuracy
    dmcsteps = int(steps_times_error2/sigma**2)+1

    scf = generate_pwscf(
        system     = system,
        job        = job(**scfjob),
        path       = path+'/scf',
        **scf_qmc_inputs,
        )

    p2q = generate_pw2qmcpack(
        identifier   = 'p2q',
        path         = path+'/scf',
        job          = job(**p2qjob),
        dependencies = [(scf,'orbitals')],
        )

    system.bconds = 'nnn'
    if jastrow is None:
        opt_cycles = 6
    else:
        opt_cycles = 3
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


# LINE-SEARCH

# 1) Surrogate: relaxation

from surrogate_tools import print_relax,get_relax_structure

relax_dir = 'relax'
run_project(get_relax_job(pos_init,path=relax_dir)) # run relax job with Nexus
# Analyze and store pos_relax for future use
pos_relax = get_relax_structure(
        path       = relax_dir,
        suffix     = 'relax.in',
        pos_units  = 'B',
        )
params_relax = pos_to_params(pos_relax) # also store the relaxed parameters
print_relax(elem,pos_relax,params_relax) # print output

# 2) Surrogate: Hessian

from surrogate_tools import load_gamma_k,compute_hessian,print_hessian_delta

phonon_dir = 'phonon'
run_project(get_phonon_jobs(pos_relax,path=phonon_dir)) # next, run phonon job

# load the real-space Hessian from the phonon calculation output
hessian_real  = load_gamma_k(phonon_dir+'/FC.fc',len(elem))
# convert to parameter hessian
hessian = compute_hessian(
        pos           = pos_relax,
        hessian_pos   = hessian_real,
        pos_to_params = pos_to_params,
        params_to_pos = params_to_pos,
        )
# obtain optimal search directions
Lambda,U    = linalg.eig(hessian)
directions  = U.T # we consider the directions transposed
print_hessian_delta(hessian,directions,Lambda) # print output

# 3) Surrogate: Optimize line-search

from matplotlib import pyplot as plt
from surrogate_error_scan import IterationData,error_scan_diagnostics,load_W_max,scan_error_data,load_of_epsilon,optimize_window_sigma,optimize_epsilond_heuristic_cost

scan_dir = 'scan_error'
scan_windows = Lambda/4
pfn = 3
pts = 7
epsilon = [0.01,0.01]

# Generate line-search object (IterationData) to manage the displacements and data
scan_data = IterationData(
        n             = 0, 
        pos           = pos_relax, 
        hessian       = hessian, 
        get_jobs      = get_scf_pes_job,
        pos_to_params = pos_to_params,
        params_to_pos = params_to_pos,
        windows       = scan_windows,
        fraction      = 0.025,
        pts           = 21,
        path          = scan_dir,
        type          = 'scf',
        load_postfix  = '/scf.in',
        colors        = ['r','b'],
        targets       = params_relax,
        )
# For later convenience: try loading the object from the disk.
#   If not there yet, generate and write. This ensures that it won't change due to statistical fluctuations.
data_load = scan_data.load_from_file()
if data_load is None:
    scan_data.shift_positions() # shift positions according to instructions
    run_project(scan_data.get_job_list()) # execute a list of Nexus jobs
    scan_data.load_results() # once executed, load results
    # estimate the maximum windows/direction that are relevant to meet the epsilon-targets
    W_max = load_W_max(
            scan_data,
            epsilon = epsilon, 
            pts     = pts,
            pfn     = pfn,
            )
    # use correlated resampling of the fits to construct a 2D-mesh of the fitting error
    scan_error_data(
            scan_data,
            pts       = pts,
            pfn       = pfn,
            generate  = 1000,
            W_num     = 16,
            W_max     = W_max, # constrain the maximum window
            sigma_num = 16,
            sigma_max = 0.1,
            )
    # based on the 2D-mesh of fitting errors, load the cost-optimal values for a range of epsilon-targets
    #   then store the trends in simple polynomial fits
    load_of_epsilon(scan_data,show_plot=True)
    # optimize the mixing of line-search errors to produce the lowest overall cost
    optimize_window_sigma(
            scan_data,
            optimizer = optimize_epsilond_heuristic_cost,
            epsilon   = epsilon,
            show_plot=True)
    # finally freeze the result by writing to file
    scan_data.write_to_file()
    plt.show()
else:
    scan_data = data_load
#end if


# print output
error_scan_diagnostics(scan_data,steps_times_error2)


# 4-5) Stochastic: Line-search

from surrogate_relax import surrogate_diagnostics,average_params,run_linesearch

# store common line-search settings
ls_settings = obj(
    get_jobs      = get_dmc_jobs,
    pos_to_params = pos_to_params,
    params_to_pos = params_to_pos,
    type          = 'qmc',
    load_postfix  = '/dmc/dmc.in.xml',
    qmc_idx       = 1,
    qmc_j_idx     = 2,
    path          = 'dmc/',
    pfn           = scan_data.pfn,
    pts           = scan_data.pts,
    windows       = scan_data.windows,
    noises        = array(scan_data.noises)/2, # from Ry to Ha
    colors        = ['r','b'],
    )

# first iteration
data = IterationData( 
        n       = 0, 
        hessian = hessian,
        pos     = pos_relax, # start from SCF relaxed position
        **ls_settings,
        )

# Run line-search up to n_max iterations, return list of line-search objects
#   run_linesearch function implements standard steps of the stochastic
#   line-search, including saving and loading restart files
data_ls = run_linesearch(data,n_max=3,ls_settings=ls_settings)

params_final, p_errs_final = average_params(
        data_ls, # input list of line-searches
        transient = 1, # take average from all steps beyond the first
        )

data_ls[0].targets = params_final
print('Final parameters:')
for p,err in zip(params_final,p_errs_final):
    print('{} +/- {}'.format(p,err))
#end for

# print and plot standard output of the iteration
surrogate_diagnostics(data_ls)
plt.show()
