#!/usr/bin/env python3

# Coronene: line-search example
#   6-parameter problem: various CC, CH and HH bond lengths
#   Surrogate theory: DFT (PBE)
#   Stochastic theory: DFT (PBE) + simulated noise
#
# This example executes a standard line-search minimization workflow for the 
# ground state of the coronene molecule, using DFT (PBE) as the surrogate
# method and DFT plus artificial noise to emulate a stochastic method.
#
# Computing task: Suitable for institutional clusters

# First, the user must set up Nexus according to their computing environment.

from nexus import generate_pwscf, generate_qmcpack, job, obj, Structure, run_project
from nexus import generate_pw2qmcpack, generate_physical_system, settings

# Modify the below variables as needed
cores      = 24
presub     = ''
qeapp      = '/path/to/pw.x'
scfjob     = obj(app=qeapp, cores=cores,ppn=cores,presub=presub,hours=2)
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


# Implement the following parametric mappings
#   p0: innermost C-C distance
#   p1: second innermost C-C distance
#   p2: second outermost C-C distance
#   p3: third C-C distance
#   p4: C-H distance
#   p5: half of H-H distance

from numpy import mean,array,sin,pi,cos,diag,linalg,arccos,arcsin

# Forward mapping: produce parameter values from an array of atomic positions
def pos_to_params(pos):
    pos = pos.reshape(-1,3) # make sure of the shape
    def distance(r1,r2):
        return sum((r1-r2)**2)**0.5
    #end def
    # for redundancy, calculate mean bond lengths
    # 0) from neighboring C-atoms
    params = array(6*[0.])
    for i in range(6):
        # list positions within one hexagon slice
        C0 = pos[6*i+0]
        C1 = pos[6*i+1]
        C2 = pos[6*i+2]
        C3 = pos[6*i+3]
        H4 = pos[6*i+4]
        H5 = pos[6*i+5]
        # get positions of neighboring slices, where necessary
        ip = 6*(i+1) % (6*6)
        im = 6*(i-1) % (6*6)
        C0ip = pos[ip+0]
        C2ip = pos[ip+2]
        C3im = pos[im+3]
        H4ip = pos[ip+4]
        # calculate parameters 
        p0 = distance(C0,C0ip)
        p1 = distance(C0,C1)
        p2 = (distance(C1,C2)+distance(C1,C3))/2
        p3 = (distance(C2ip,C3)+distance(C2,C3im))/2
        p4 = (distance(C2,H4)+distance(C3,H5))/2
        p5 = distance(H4ip,H5)/2
        params += array([p0,p1,p2,p3,p4,p5])/6
    #end for
    return params
#end def

# Backward mapping: produce array of atomic positions from parameters
axes = array([30,30,10]) # simulate in vacuum
def params_to_pos(params):
    p0,p1,p2,p3,p4,p5 = tuple(params)

    # define 2D rotation matrix in the xy plane
    def rotate_xy(angle):
        return array([[cos(angle),-sin(angle),0.0],
                      [sin(angle), cos(angle),0.0],
                      [0.0       , 0.0       ,1.0]])
    #end def

    # auxiliary geometrical variables
    y1    = sin(pi/6)*(p0+p1)-p3/2
    alpha = arccos(y1/p2)
    x1    = (p0+p1)*cos(pi/6)+p2*sin(alpha)
    beta  = arcsin((p5-p3/2)/p4)
    x2    = x1+p4*cos(beta)

    # closed forms for the atomic positions with the aux variables
    C0 = array([p0,0.,0.])      @ rotate_xy(-pi/6)
    C1 = array([(p0+p1),0.,0.]) @ rotate_xy(-pi/6)
    C2 = array([x1, p3/2, 0.0])
    C3 = array([x1,-p3/2, 0.0]) @ rotate_xy(-pi/3) 
    H4 = array([x2, p5, 0.0])
    H5 = array([x2,-p5, 0.0])   @ rotate_xy(-pi/3) 

    pos = []
    for i in range(6):
        ang = i*pi/3
        pos.append( rotate_xy(ang) @ C0 )
        pos.append( rotate_xy(ang) @ C1 )
        pos.append( rotate_xy(ang) @ C2 )
        pos.append( rotate_xy(ang) @ C3 )
        pos.append( rotate_xy(ang) @ H4 )
        pos.append( rotate_xy(ang) @ H5 )
    #end for

    pos = (axes/2 + array(pos)).flatten()
    return pos
#end def

# Guess initial parameter values
p_init = array([2.69,2.69, 2.69, 2.60, 2.07, 2.34])
pos_init = params_to_pos(p_init)
elem = 6*(4*['C']+2*['H'])

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

# construct Nexus system based on position
valences = obj(C=4,H=1) # pseudo-valences

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
    return generate_physical_system(structure=structure,**valences)
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
phjob    = obj(app_command='ph.x -in phonon.in', cores=cores,ppn=cores,presub=presub,hours=2,mem=380)
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
from surrogate_error_scan import IterationData,error_scan_diagnostics,load_W_max,scan_error_data,load_of_epsilon,optimize_window_sigma,optimize_epsilond_thermal

scan_dir = 'scan_error'
scan_windows = Lambda/4
pfn = 3
pts = 7
epsilon = 6*[0.01]

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
        pts           = 15,
        path          = scan_dir,
        type          = 'scf',
        load_postfix  = '/scf.in',
        colors        = ['r','b','c','g','m','k'],
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
            sigma_max = 0.05,
            )
    # based on the 2D-mesh of fitting errors, load the cost-optimal values for a range of epsilon-targets
    #   then store the trends in simple polynomial fits
    load_of_epsilon(scan_data,show_plot=True)
    # optimize the mixing of line-search errors to produce the lowest overall cost
    optimize_window_sigma(
            scan_data,
            optimizer = optimize_epsilond_thermal,
            epsilon   = epsilon,
            show_plot=True)
    # finally freeze the result by writing to file
    scan_data.write_to_file()
    plt.show()
else:
    scan_data = data_load
#end if

# print output
error_scan_diagnostics(scan_data)


# 4-5) Stochastic: Line-search

from surrogate_relax import surrogate_diagnostics,average_params

# store common line-search settings
ls_settings = obj(
    get_jobs      = get_scf_pes_job,
    pos_to_params = pos_to_params,
    params_to_pos = params_to_pos,
    type          = 'scf',
    load_postfix  = '/scf.in',
    path          = 'scf+noise/',
    pfn           = scan_data.pfn,
    pts           = scan_data.pts,
    windows       = scan_data.windows,
    noises        = scan_data.noises,
    colors        = ['r','b','c','g','m','k'],
    targets       = params_relax, # real targets are known
    )

# Start from a displaced position and converge back to equilibrium
params_init = params_relax + array([0.05,-0.05,0.05,-0.05,0.05,-0.05])
pos_init = params_to_pos(params_init)
data = IterationData( 
        n       = 0, 
        hessian = hessian,
        pos     = pos_init,
        **ls_settings,
        )

# Run line-search up to n_max iterations, return list of line-search objects
#   run_linesearch function implements standard steps of the stochastic
#   line-search, including saving and loading restart files
data_ls = run_linesearch(data,n_max=4,ls_settings=ls_settings)

params_final, p_errs_final = average_params(
        data_ls, # input list of line-searches
        transient = 0, # take average from all steps beyond the first
        )

print('Final parameters:')
for p,err in zip(params_final,p_errs_final):
    print('{} +/- {}'.format(p,err))
#end for

# print and plot standard output of the iteration
surrogate_diagnostics(data_ls)
plt.show()
