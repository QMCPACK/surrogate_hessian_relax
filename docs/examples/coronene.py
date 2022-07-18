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
from nexus import generate_pwscf, generate_qmcpack, job, obj 
from nexus import generate_pw2qmcpack, generate_physical_system

# Modify the below variables as needed
base_dir   = 'coronene/'
cores      = 8
presub     = ''
qeapp      = 'pw.x'
p2qapp     = 'pw2qmcpack.x'
qmcapp     = 'qmcpack'
scfjob     = obj(app=qeapp, cores=cores,ppn=cores,presub=presub)
p2qjob     = obj(app=p2qapp,cores=1,ppn=1,presub=presub)
optjob     = obj(app=qmcapp,cores=cores,ppn=cores,presub=presub)
dmcjob     = obj(app=qmcapp,cores=cores,ppn=cores,presub=presub)
nx_settings = obj(
    sleep         = 3,
    pseudo_dir    = 'pseudos',
    runs          = '',
    results       = '',
    status_only   = 0,
    generate_only = 0,
    machine       = 'ws8',
    )
from surrogate_macros import init_nexus
init_nexus(**nx_settings) # initiate nexus
fake_sim = not __name__=='__main__'  # this is needed for object serialization

# Pseudos (execute download_pseudos.sh in the working directory)
relaxpseudos   = ['C.pbe_v1.2.uspp.F.upf', 'H.pbe_v1.4.uspp.F.upf']
qmcpseudos     = ['C.ccECP.xml','H.ccECP.xml']
scfpseudos     = ['C.ccECP.upf','H.ccECP.upf']


# Implement the following parametric mappings for benzene
#   p0: C-C distance
#   p1: C-H distance

from numpy import mean,array,sin,pi,cos,diag,linalg,arccos,arcsin
from surrogate_tools import distance

# Forward mapping: produce parameter values from an array of atomic positions
def forward(pos, **kwargs):
    pos = pos.reshape(-1, 3) # make sure of the shape
    # 0) from neighboring C-atoms
    params = array(6 * [0.])
    for i in range(6):
        # list positions within one hexagon slice
        C0 = pos[6 * i + 0]
        C1 = pos[6 * i + 1]
        C2 = pos[6 * i + 2]
        C3 = pos[6 * i + 3]
        H4 = pos[6 * i + 4]
        H5 = pos[6 * i + 5]
        # get positions of neighboring slices, where necessary
        ip = 6 * (i + 1) % (6 * 6)
        im = 6 * (i - 1) % (6 * 6)
        C0ip = pos[ip + 0]
        C2ip = pos[ip + 2]
        C3im = pos[im + 3]
        H4ip = pos[ip + 4]
        # calculate parameters 
        p0 = distance(C0, C0ip)
        p1 = distance(C0, C1)
        p2 = (distance(C1, C2) + distance(C1, C3)) / 2
        p3 = (distance(C2ip, C3) + distance(C2, C3im)) / 2
        p4 = (distance(C2, H4) + distance(C3, H5)) / 2
        p5 = distance(H4ip, H5) / 2
        params += array([p0, p1, p2, p3, p4, p5]) / 6
    #end for
    return params
#end def


# Backward mapping: produce array of atomic positions from parameters
axes = array([30, 30, 10]) # simulate in vacuum
def backward(params, **kwargs):
    p0, p1, p2, p3, p4, p5 = tuple(params)
    # define 2D rotation matrix in the xy plane
    def rotate_xy(angle):
        return array([[cos(angle),-sin(angle), 0.0],
                      [sin(angle), cos(angle), 0.0],
                      [0.0       , 0.0       , 1.0]])
    #end def
    # auxiliary geometrical variables
    y1    = sin(pi / 6) * (p0 + p1) - p3 / 2
    alpha = arccos(y1 / p2)
    x1    = (p0 + p1) * cos(pi / 6) + p2 * sin(alpha)
    beta  = arcsin((p5 - p3 / 2) / p4)
    x2    = x1 + p4 * cos(beta)
    # closed forms for the atomic positions with the aux variables
    C0 = array([p0, 0., 0.])      @ rotate_xy(-pi / 6)
    C1 = array([(p0+p1), 0., 0.]) @ rotate_xy(-pi / 6)
    C2 = array([x1, p3/2, 0.0])
    C3 = array([x1, -p3/2, 0.0])  @ rotate_xy(-pi / 3)
    H4 = array([x2, p5, 0.0])
    H5 = array([x2, -p5, 0.0])    @ rotate_xy(-pi / 3)
    pos = []
    for i in range(6):
        ang = i * pi / 3
        pos.append(rotate_xy(ang) @ C0)
        pos.append(rotate_xy(ang) @ C1)
        pos.append(rotate_xy(ang) @ C2)
        pos.append(rotate_xy(ang) @ C3)
        pos.append(rotate_xy(ang) @ H4)
        pos.append(rotate_xy(ang) @ H5)
    #end for
    pos = (axes/2 + array(pos)).flatten()
    return pos
#end def


# Let us initiate a ParameterStructure object that conforms to the above mappings
from surrogate_classes import ParameterStructure
params_init = array([2.69,2.69, 2.69, 2.60, 2.07, 2.34])
elem = 6 * (4 * ['C'] + 2 * ['H'])
structure_init = ParameterStructure(
    forward = forward,
    backward = backward,
    params = params_init,
    elem = elem,
    kgrid = (1, 1, 1),  # needed to run plane-waves with Nexus
    kshift = (0, 0, 0,),
    dim = 3,
    units = 'B',
    axes = diag(axes),
    periodic = False)

# return a 1-item list of Nexus jobs: SCF relaxation
def scf_relax_job(structure, path, **kwargs):
    system = generate_physical_system(structure = structure, C = 4, H = 1)
    relax = generate_pwscf(
        system        = system,
        job           = job(**scfjob),
        path          = path,
        pseudos       = relaxpseudos,
        identifier    = 'relax',
        calculation   = 'relax',
        input_type    = 'generic',
        input_dft     = 'pbe',
        occupations   = 'smearing',
        smearing      = 'gaussian',
        degauss       = 0.0001,
        nosym         = False,
        conv_thr      = 1e-9,
        mixing_beta   = .7,
        electron_maxstep = 1000,
        ecutwfc       = 100,
        ecutrho       = 300,
        forc_conv_thr = 1e-4,
        ion_dynamics  = 'bfgs',
        fake_sim      = fake_sim,
    )
    return [relax]
#end def


# LINE-SEARCH

# 1) Surrogate: relaxation

# Let us run a macro that computes and returns the relaxed structure
from surrogate_macros import relax_structure
structure_relax = relax_structure(
    structure = structure_init, 
    relax_job = scf_relax_job,
    path = base_dir + 'relax/'
)


# 2 ) Surrogate: Hessian

# Let us use phonon calculation to obtain the surrogate Hessian
# First, define a 3-item list of Nexus jobs: 
#   1: SCF single-shot calculation
#   2: SCF phonon calculation
#   3: Conversion to force-constant matrix
# Since the phonon calculations are not standard in Nexus, we are providing the 
# inputs manually by using GenericSimulation and input_template classes
from simulation import GenericSimulation, input_template
phjob    = obj(app_command = 'ph.x -in phonon.in', cores = cores, ppn = cores, presub = presub, hours = 2)
q2rjob   = obj(app_command = 'q2r.x -in q2r.in', cores = cores, ppn = cores, presub = presub, hours = 2)
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
def get_phonon_jobs(structure, path, **kwargs):
    system = generate_physical_system(structure = structure, C = 4, H = 1)
    scf = generate_pwscf(
        system     = system,
        job        = job(**scfjob),
        path       = path,
        pseudos       = relaxpseudos,
        identifier    = 'scf',
        calculation   = 'scf',
        input_type    = 'generic',
        input_dft     = 'pbe',
        occupations   = 'smearing',
        smearing      = 'gaussian',
        degauss       = 0.0001,
        nosym         = False,
        nogamma       = True,
        mixing_beta   = .7,
        ecutwfc       = 100,
        ecutrho       = 300,
        electron_maxstep = 1000,
        fake_sim      = fake_sim,
        )
    scf.input.control.disk_io = 'low' # write charge density
    phonon = GenericSimulation(
        system       = system,
        job          = job(**phjob),
        path         = path,
        input        = ph_input,
        identifier   = 'phonon',
        fake_sim     = fake_sim,
        dependencies = (scf, 'other'),
        )
    q2r = GenericSimulation(
        system       = system,
        job          = job(**q2rjob),
        path         = path,
        input        = q2r_input,
        identifier   = 'q2r',
        fake_sim     = fake_sim,
        dependencies = (phonon, 'other'),
        )
    return [scf, phonon, q2r]
#end def

# Finally, use a macro to read the phonon data and convert to parameter
# Hessian based on the structural mappings
from surrogate_macros import compute_phonon_hessian
hessian = compute_phonon_hessian(
    structure = structure_relax,
    phonon_job = get_phonon_jobs,
    path = base_dir + 'phonon'
)


# 3) Surrogate: Optimize line-search

# Let us define an SCF PES job that is consistent with the earlier relaxation
def scf_pes_job(structure, path, **kwargs):
    system = generate_physical_system(
        structure = structure,
        C = 4,
        H = 1,
    )
    scf = generate_pwscf(
        system        = system,
        job           = job(**scfjob),
        path          = path,
        pseudos       = relaxpseudos,
        identifier    = 'scf',
        calculation   = 'scf',
        input_type    = 'generic',
        input_dft     = 'pbe',
        occupations   = 'smearing',
        smearing      = 'gaussian',
        degauss       = 0.0001,
        nosym         = False,
        mixing_beta   = .7,
        forc_conv_thr = 1e-4,
        ecutwfc       = 100,
        ecutrho       = 300,
        electron_maxstep = 1000,
        fake_sim      = fake_sim,
    )
    return [scf]
#end def

# Use a macro to generate a parallel line-search object that samples the
# surrogate PES around the minimum along the search directions
from surrogate_macros import generate_surrogate, plot_surrogate_pes, plot_surrogate_bias
from matplotlib import pyplot as plt
surrogate = generate_surrogate(
    structure = structure_relax,
    hessian = hessian,
    func = scf_pes_job,
    path = base_dir + 'surrogate/',  
    window_frac = 0.25,  # maximum displacement relative to Lambda of each direction
    noise_frac = 0.1,  # (initial) maximum resampled noise relative to the maximum window
    load = 'surrogate.p',  # try to load from disk
    M = 15)  # number of points per direction to sample (should be more than finally intended)

bias_mix = 0.1  # how much energy displacement is mixed in to assess the total bias (default: 0.0)
M = 7  # how many points in the final line-search
# diagnose: plot bias using different fit kinds
if __name__=='__main__':
    plot_surrogate_pes(surrogate)
    plot_surrogate_bias(surrogate, fit_kind = 'pf2', M = M, bias_mix = bias_mix)
    plot_surrogate_bias(surrogate, fit_kind = 'pf3', M = M, bias_mix = bias_mix)
    plot_surrogate_bias(surrogate, fit_kind = 'pf4', M = M, bias_mix = bias_mix)
    plt.show()
#end if

# Set target parameter error tolerances (epsilon): 0.01 Bohr accuracy for both C-C and C-H bonds.
# Then, optimize the surrogate line-search to meet the tolerances given the line-search
#   main input: bias_mix, M, epsilon
#   main output: windows, noises (per direction to meet all epsilon)
epsilon = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
from surrogate_macros import optimize_surrogate
optimize_surrogate(
    surrogate,  # provide the surrogate object
    epsilon = epsilon,
    fit_kind = 'pf3',
    bias_mix = bias_mix,
    M = M,
    N = 400,  # use as many points for correlated resampling of the error
    save = 'surrogate.p',  # serialize the object to freeze the stochastic process
    rewrite = False,)  # don't rewrite old work, if present

# Diagnoze and plot the performance of the surrogate optimization
from surrogate_macros import surrogate_diagnostics
surrogate_diagnostics(surrogate)
if __name__=='__main__':
    plt.show()
#end if

# The check (optional) the performance, let us simulate a line-search on the surrogate PES.
# It is cheaper to debug the optimizer here than later on.
# First, generate line-search iteration object based on the surrogate
from surrogate_macros import generate_linesearch, propagate_linesearch, nexus_pwscf_analyzer
srg_ls = generate_linesearch(
    surrogate,
    job_func = scf_pes_job,  # use the surrogate PES
    path = base_dir + 'srg_ls/',
    analyze_func = nexus_pwscf_analyzer,  # use this method to read the data
    shift_params = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1],  # shift the starting parameters
    load = True,  # try loading the object, if present
    load_only = fake_sim,  # WIP: override import problems
)
# Propagate the parallel line-search (compute values, analyze, then move on) 4 times
#   add_sigma = True means that target errorbars are used to simulate random noise
propagate_linesearch(srg_ls, i = 0, add_sigma = True)
propagate_linesearch(srg_ls, i = 1, add_sigma = True)
propagate_linesearch(srg_ls, i = 2, add_sigma = True)
propagate_linesearch(srg_ls, i = 3, add_sigma = True)

# Diagnoze and plot the line-search performance.
from surrogate_macros import linesearch_diagnostics
linesearch_diagnostics(srg_ls)
if __name__=='__main__':
    plt.show()
#end if


# 4-5) Stochastic: Line-search


# return a simple 4-item DMC workflow for Nexus:
#   1: run SCF with norm-conserving ECPs to get orbitals
#   2: convert for QMCPACK
#   3: Optimize 2-body Jastrow coefficients
#   4: Run DMC with enough steps/block to meet the target errorbar sigma
#     it is important to first characterize the DMC run into var_eff
from surrogate_macros import dmc_steps
def dmc_pes_job(structure, path, sigma = 0.01, var_eff = 1.0, **kwargs):
    system = generate_physical_system(
        structure = structure,
        C = 4,
        H = 1,
    )
    scf = generate_pwscf(
        system        = system,
        job           = job(**scfjob),
        path          = path+'/scf',
        pseudos       = scfpseudos,
        identifier    = 'scf',
        calculation   = 'scf',
        input_type    = 'generic',
        input_dft     = 'pbe',
        nosym         = False,
        nogamma       = True,
        conv_thr      = 1e-9,
        mixing_beta   = .7,
        ecutwfc       = 300,
        ecutrho       = 600,
        occupations   = 'smearing',
        smearing      = 'gaussian',
        degauss       = 0.0001,
        electron_maxstep = 1000,
        fake_sim      = fake_sim,
    )
    p2q = generate_pw2qmcpack(
        identifier   = 'p2q',
        path         = path+'/scf',
        job          = job(**p2qjob),
        dependencies = [(scf,'orbitals')],
        fake_sim     = fake_sim,
    )
    system.bconds = 'nnn'
    opt = generate_qmcpack(
        system       = system,
        path         = path+'/opt',
        job          = job(**optjob),
        dependencies = [(p2q,'orbitals')],
        cycles       = 8,
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
        fake_sim     = fake_sim,
        )
    dmcsteps = dmc_steps(sigma, var_eff = var_eff)
    dmc = generate_qmcpack(
        system       = system,
        path         = path+'/dmc',
        job          = job(**dmcjob),
        dependencies = [(p2q,'orbitals'),(opt,'jastrow') ],
        steps        = dmcsteps,
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
        fake_sim     = fake_sim,
        )
    return [scf,p2q,opt,dmc]
#end def

# Run a macro that runs a DMC test job and returns effective variance w.r.t the number of steps/block
from surrogate_macros import get_var_eff
var_eff = get_var_eff(
    structure_relax,
    dmc_pes_job,
    path = base_dir + 'dmc_test',
    suffix = '/dmc/dmc.in.xml',
)

# Finally, use a macro to generate a parallel line-seach iteration object based on the DMC PES
from surrogate_macros import nexus_qmcpack_analyzer
dmc_ls = generate_linesearch(
    surrogate,
    job_func = dmc_pes_job,
    path = base_dir + 'dmc_ls/',
    load = True,
    load_only = fake_sim,  # WIP: override import problems
    job_args = {'var_eff': var_eff},  # provide DMC job with var_eff
    c_noises = 0.5,  # WIP: convert noises from Ry (SCF) to Ha (QMCPACK)
    analyze_func = nexus_qmcpack_analyzer,  # use this function to analyze the energies
)
# Propagate the line-search 3 times
propagate_linesearch(dmc_ls, i = 0)
propagate_linesearch(dmc_ls, i = 1)
propagate_linesearch(dmc_ls, i = 2)

# Diagnoze and plot the line-search performance
from surrogate_macros import linesearch_diagnostics
linesearch_diagnostics(dmc_ls)
if __name__=='__main__':
    plt.show()
#end if
