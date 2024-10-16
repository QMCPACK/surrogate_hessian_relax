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
from surrogate_macros import nexus_qmcpack_analyzer
from surrogate_macros import get_var_eff
from surrogate_macros import dmc_steps
from surrogate_macros import linesearch_diagnostics
from surrogate_macros import propagate_linesearch
from surrogate_classes import LineSearchIteration
from surrogate_macros import surrogate_diagnostics
from matplotlib import pyplot as plt
from surrogate_macros import nexus_pwscf_analyzer
from surrogate_macros import generate_surrogate, plot_surrogate_pes, plot_surrogate_bias
from surrogate_macros import compute_phonon_hessian
from simulation import GenericSimulation, input_template
from surrogate_macros import relax_structure
from surrogate_classes import ParameterStructure
from numpy import mean, array, sin, pi, cos, diag, linalg
from nexus import generate_pwscf, generate_qmcpack, job, obj
from nexus import generate_pw2qmcpack, generate_physical_system
from nxs import scfjob, p2qjob, optjob, dmcjob, cores, presub

# Pseudos (execute download_pseudos.sh in the working directory)
base_dir = 'benzene/'
interactive = False  # want to be interactive or not?
relaxpseudos = ['C.pbe_v1.2.uspp.F.upf', 'H.pbe_v1.4.uspp.F.upf']
qmcpseudos = ['C.ccECP.xml', 'H.ccECP.xml']
scfpseudos = ['C.ccECP.upf', 'H.ccECP.upf']

# Implement the following parametric mappings for benzene
#   p0: C-C distance
#   p1: C-H distance


# Forward mapping: produce parameter values from an array of atomic positions

def forward(pos, **kwargs):
    pos = pos.reshape(-1, 3)  # make sure of the shape
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

    def distance(r1, r2):
        return sum((r1 - r2)**2)**0.5
    # end def
    # for redundancy, calculate mean bond lengths
    # 0) from neighboring C-atoms
    r_CC = mean([distance(C0, C1),
                 distance(C1, C2),
                 distance(C2, C3),
                 distance(C3, C4),
                 distance(C4, C5),
                 distance(C5, C0)])
    # 1) from corresponding H-atoms
    r_CH = mean([distance(C0, H0),
                 distance(C1, H1),
                 distance(C2, H2),
                 distance(C3, H3),
                 distance(C4, H4),
                 distance(C5, H5)])
    params = array([r_CC, r_CH])
    return params
# end def


# Backward mapping: produce array of atomic positions from parameters
axes = array([20, 20, 10])  # simulate in vacuum


def backward(params, **kwargs):
    r_CC = params[0]
    r_CH = params[1]
    # place atoms on a hexagon in the xy-directions
    hex_xy = array([[cos(3 * pi / 6), sin(3 * pi / 6), 0.],
                    [cos(5 * pi / 6), sin(5 * pi / 6), 0.],
                    [cos(7 * pi / 6), sin(7 * pi / 6), 0.],
                    [cos(9 * pi / 6), sin(9 * pi / 6), 0.],
                    [cos(11 * pi / 6), sin(11 * pi / 6), 0.],
                    [cos(13 * pi / 6), sin(13 * pi / 6), 0.]])
    pos_C = axes / 2 + hex_xy * r_CC  # C-atoms are one C-C length apart from origin
    # H-atoms one C-H length apart from C-atoms
    pos_H = axes / 2 + hex_xy * (r_CC + r_CH)
    pos = array([pos_C, pos_H]).flatten()
    return pos
# end def


# Let us initiate a ParameterStructure object that conforms to the above mappings
params_init = array([2.651, 2.055])
elem = 6 * ['C'] + 6 * ['H']
structure_init = ParameterStructure(
    forward=forward,
    backward=backward,
    params=params_init,
    elem=elem,
    kgrid=(1, 1, 1),  # needed to run plane-waves with Nexus
    kshift=(0, 0, 0,),
    dim=3,
    units='B',
    axes=diag(axes),
    periodic=False)

# return a 1-item list of Nexus jobs: SCF relaxation


def scf_relax_job(structure, path, **kwargs):
    system = generate_physical_system(structure=structure, C=4, H=1)
    relax = generate_pwscf(
        system=system,
        job=job(**scfjob),
        path=path,
        pseudos=relaxpseudos,
        identifier='relax',
        calculation='relax',
        input_type='generic',
        input_dft='pbe',
        occupations='smearing',
        smearing='gaussian',
        degauss=0.0001,
        nosym=False,
        conv_thr=1e-9,
        mixing_beta=.7,
        electron_maxstep=1000,
        ecutwfc=100,
        ecutrho=300,
        forc_conv_thr=1e-4,
        ion_dynamics='bfgs',
    )
    return [relax]
# end def


# LINE-SEARCH

# 1) Surrogate: relaxation

# Let us run a macro that computes and returns the relaxed structure
structure_relax = relax_structure(
    structure=structure_init,
    relax_job=scf_relax_job,
    path=base_dir + 'relax/'
)

# 2 ) Surrogate: Hessian

# Let us use phonon calculation to obtain the surrogate Hessian
# First, define a 3-item list of Nexus jobs:
#   1: SCF single-shot calculation
#   2: SCF phonon calculation
#   3: Conversion to force-constant matrix
# Since the phonon calculations are not standard in Nexus, we are providing the
# inputs manually by using GenericSimulation and input_template classes
phjob = obj(app_command='ph.x -in phonon.in', cores=cores,
            ppn=cores, presub=presub, hours=2)
q2rjob = obj(app_command='q2r.x -in q2r.in', cores=cores,
             ppn=cores, presub=presub, hours=2)
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
    system = generate_physical_system(structure=structure, C=4, H=1)
    scf = generate_pwscf(
        system=system,
        job=job(**scfjob),
        path=path,
        pseudos=relaxpseudos,
        identifier='scf',
        calculation='scf',
        input_type='generic',
        input_dft='pbe',
        occupations='smearing',
        smearing='gaussian',
        degauss=0.0001,
        nosym=False,
        nogamma=True,
        mixing_beta=.7,
        ecutwfc=100,
        ecutrho=300,
        electron_maxstep=1000,
    )
    scf.input.control.disk_io = 'low'  # write charge density
    phonon = GenericSimulation(
        system=system,
        job=job(**phjob),
        path=path,
        input=ph_input,
        identifier='phonon',
        dependencies=(scf, 'other'),
    )
    q2r = GenericSimulation(
        system=system,
        job=job(**q2rjob),
        path=path,
        input=q2r_input,
        identifier='q2r',
        dependencies=(phonon, 'other'),
    )
    return [scf, phonon, q2r]
# end def


# Finally, use a macro to read the phonon data and convert to parameter
# Hessian based on the structural mappings
hessian = compute_phonon_hessian(
    structure=structure_relax,
    phonon_job=get_phonon_jobs,
    path=base_dir + 'phonon',
)


# 3) Surrogate: Optimize line-search

# Let us define an SCF PES job that is consistent with the earlier relaxation
def scf_pes_job(structure, path, **kwargs):
    system = generate_physical_system(
        structure=structure,
        C=4,
        H=1,
    )
    scf = generate_pwscf(
        system=system,
        job=job(**scfjob),
        path=path,
        pseudos=relaxpseudos,
        identifier='scf',
        calculation='scf',
        input_type='generic',
        input_dft='pbe',
        occupations='smearing',
        smearing='gaussian',
        degauss=0.0001,
        nosym=False,
        mixing_beta=.7,
        forc_conv_thr=1e-4,
        ecutwfc=100,
        ecutrho=300,
        electron_maxstep=1000,
    )
    return [scf]
# end def


# Use a macro to generate a parallel line-search object that samples the
# surrogate PES around the minimum along the search directions
surrogate = generate_surrogate(
    path=base_dir + 'surrogate/',
    fname='surrogate.p',  # try to load from disk
    structure=structure_relax,
    hessian=hessian,
    pes_func=scf_pes_job,
    load_func=nexus_pwscf_analyzer,
    mode='nexus',
    window_frac=0.25,  # maximum displacement relative to Lambda of each direction
    # number of points per direction to sample (should be more than finally intended)
    M=15)
surrogate.run_jobs(interactive=interactive)
surrogate.load_results(set_target=True)

# diagnose: plot bias using different fit kinds
if __name__ == '__main__' and interactive:
    plot_surrogate_pes(surrogate)
    plot_surrogate_bias(surrogate, fit_kind='pf2', M=7)
    plot_surrogate_bias(surrogate, fit_kind='pf3', M=7)
    plot_surrogate_bias(surrogate, fit_kind='pf4', M=7)
    plt.show()
# end if

# Set target parameter error tolerances (epsilon): 0.01 Bohr accuracy for both C-C and C-H bonds.
# Then, optimize the surrogate line-search to meet the tolerances given the line-search
#   main input: M, epsilon
#   main output: windows, noises (per direction to meet all epsilon)
epsilon_p = [0.02, 0.02]
if not surrogate.optimized:
    surrogate.optimize(
        epsilon_p=epsilon_p,
        fit_kind='pf4',
        # (initial) maximum resampled noise relative to the maximum window
        noise_frac=0.1,
        M=7,
        N=400,  # use as many points for correlated resampling of the error
    )
    surrogate.write_to_disk('surrogate.p')
# end if

# Diagnose and plot the performance of the surrogate optimization
if __name__ == '__main__' and interactive:
    surrogate_diagnostics(surrogate)
    plt.show()
# end if

# The check (optional) the performance, let us simulate a line-search on the surrogate PES.
# It is cheaper to debug the optimizer here than later on.
# First, shift parameters for the show
surrogate_shifted = surrogate.copy()
surrogate_shifted.structure.shift_params([0.1, -0.1])
# Then generate line-search iteration object based on the shifted surrogate
srg_ls = LineSearchIteration(
    surrogate=surrogate_shifted,
    mode='nexus',
    path=base_dir + 'srg_ls',
    pes_func=scf_pes_job,  # use the surrogate PES
    load_func=nexus_pwscf_analyzer,  # use this method to read the data
)
# Propagate the parallel line-search (compute values, analyze, then move on) 4 times
#   add_sigma = True means that target errorbars are used to simulate random noise
for i in range(4):
    srg_ls.pls(i).run_jobs(interactive=interactive)
    srg_ls.pls(i).load_results(add_sigma=True)
    srg_ls.propagate(i)
# end for
srg_ls.pls(4).run_jobs(interactive=interactive, eqm_only=True)
srg_ls.pls(4).load_eqm_results(add_sigma=True)
# Diagnose and plot the line-search performance.
if __name__ == '__main__' and interactive:
    linesearch_diagnostics(srg_ls)
    plt.show()
# end if


# 4-5) Stochastic: Line-search


# return a simple 4-item DMC workflow for Nexus:
#   1: run SCF with norm-conserving ECPs to get orbitals
#   2: convert for QMCPACK
#   3: Optimize 2-body Jastrow coefficients
#   4: Run DMC with enough steps/block to meet the target errorbar sigma
#     it is important to first characterize the DMC run into var_eff


def dmc_pes_job(structure, path, sigma=0.01, var_eff=1.0, **kwargs):
    system = generate_physical_system(
        structure=structure,
        C=4,
        H=1,
    )
    scf = generate_pwscf(
        system=system,
        job=job(**scfjob),
        path=path + '/scf',
        pseudos=scfpseudos,
        identifier='scf',
        calculation='scf',
        input_type='generic',
        input_dft='pbe',
        nosym=False,
        nogamma=True,
        conv_thr=1e-9,
        mixing_beta=.7,
        ecutwfc=300,
        ecutrho=600,
        occupations='smearing',
        smearing='gaussian',
        degauss=0.0001,
        electron_maxstep=1000,
    )
    p2q = generate_pw2qmcpack(
        identifier='p2q',
        path=path + '/scf',
        job=job(**p2qjob),
        dependencies=[(scf, 'orbitals')],
    )
    system.bconds = 'nnn'
    opt = generate_qmcpack(
        system=system,
        path=path + '/opt',
        job=job(**optjob),
        dependencies=[(p2q, 'orbitals')],
        cycles=8,
        identifier='opt',
        qmc='opt',
        input_type='basic',
        pseudos=qmcpseudos,
        bconds='nnn',
        J2=True,
        J1_size=6,
        J1_rcut=6.0,
        J2_size=8,
        J2_rcut=8.0,
        minmethod='oneshift',
        blocks=200,
        substeps=2,
        steps=1,
        samples=100000,
        minwalkers=0.1,
        nonlocalpp=True,
        use_nonlocalpp_deriv=False,
    )
    dmcsteps = dmc_steps(sigma, var_eff=var_eff)
    dmc = generate_qmcpack(
        system=system,
        path=path + '/dmc',
        job=job(**dmcjob),
        dependencies=[(p2q, 'orbitals'), (opt, 'jastrow')],
        steps=dmcsteps,
        identifier='dmc',
        qmc='dmc',
        input_type='basic',
        pseudos=qmcpseudos,
        bconds='nnn',
        jastrows=[],
        vmc_samples=1000,
        blocks=200,
        timestep=0.01,
        nonlocalmoves=True,
        ntimesteps=1,
    )
    return [scf, p2q, opt, dmc]
# end def


# Run a macro that runs a DMC test job and returns effective variance w.r.t the number of steps/block
var_eff = get_var_eff(
    structure_relax,
    dmc_pes_job,
    path=base_dir + 'dmc_test',
    suffix='/dmc/dmc.in.xml',
)

# Finally, use a macro to generate a parallel line-seach iteration object based on the DMC PES
dmc_ls = LineSearchIteration(
    surrogate=surrogate,
    c_noises=0.5,  # WIP: convert noises from Ry (SCF) to Ha (QMCPACK)
    mode='nexus',
    path=base_dir + 'dmc_ls',
    pes_func=dmc_pes_job,
    pes_args={'var_eff': var_eff},  # provide DMC job with var_eff
    load_func=nexus_qmcpack_analyzer,
    load_args={'suffix': 'dmc/dmc.in.xml', 'qmc_id': 1},
)
for i in range(3):
    dmc_ls.pls(i).run_jobs(interactive=interactive)
    dmc_ls.pls(i).load_results()
    dmc_ls.propagate(i)
# end for
dmc_ls.pls(3).run_jobs(interactive=interactive, eqm_only=True)
dmc_ls.pls(3).load_eqm_results()

# Diagnose and plot the line-search performance
if __name__ == '__main__' and interactive:
    linesearch_diagnostics(dmc_ls)
    plt.show()
# end if
