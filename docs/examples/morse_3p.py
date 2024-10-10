#!/usr/bin/env python3

# Morse 3p: line-search example
#   3-parameter problem in the abstract parameter space
#
# This example simulates characteristics of the line-search method using
# a lightweight and artificial Morse PES.
#
# Computing task: Runs on command line


# from Nexus
from surrogate_macros import linesearch_diagnostics
from surrogate_classes import LineSearchIteration
from surrogate_macros import generate_surrogate
from surrogate_macros import compute_fdiff_hessian
from scipy.optimize import minimize
from surrogate_classes import ParameterSet
base_dir = 'morse_3p/'

# takes: a structure object with an attribute 3x1 array params, also target noise (sigma)
#   c: coupling constant of parameters through auxiliary morse potentials
#   d: eqm displacements
# returns: energy value, error (= sigma)


def pes(structure, sigma=None, c=1.0, d=0.0):
    from numpy import array, exp, random

    def morse(p, r):
        return p[2] * ((1 - exp(-(r - p[0]) / p[1]))**2 - 1) + p[3]

    p0, p1, p2 = structure.params
    # define Morse potentials for each individual parameter
    # when c = 0, these are the solutions for p_eqm and the Hessian
    #  (eqm value, stiffness, well depth, E_inf)
    m0 = array([1.0 + d, 3.0, 0.5, 0.0])
    m1 = array([2.0 + d, 2.0, 0.5, 0.0])
    m2 = array([3.0 + d, 1.0, 0.5, 0.0])
    E = 0.0
    E += morse(m0, p0)
    E += morse(m1, p1)
    E += morse(m2, p2)
    # non-zero (c > 0) couplings between the parameters set off the equilibrium point
    m01 = array([4.0, 6.0, 0.5, 0.0])
    m02 = array([5.0, 5.0, 0.5, 0.0])
    m12 = array([6.0, 4.0, 0.5, 0.0])
    E += c * morse(m01, p0 + p1)
    E += c * morse(m02, p0 + p2)
    E += c * morse(m12, p1 + p2)
    if sigma is not None:
        E += sigma * random.randn(1)[0]  # add random noise
    # end if
    return E, sigma
# end def


# Guess the initial structure based on the non-coupled equilibria
p_init = ParameterSet([1.0, 2.0, 3.0])
c_srg = 1.0  # define the surrogate PES with c = 1.0
d_srg = 0.0  # and d = 0.0

# Relax numerically in the absence of noise
# wrap the function for numerical optimizer


def pes_min(params):
    return pes(ParameterSet(params), c=c_srg, d=d_srg)[0]


# end def
res = minimize(pes_min, p_init.params)
p_relax = ParameterSet(res.x)
print('Minimum-energy parameters (surrogate):')
print(p_relax.params)

# Compute the numerical Hessian using a finite difference method
hessian = compute_fdiff_hessian(structure=p_relax, func=pes, mode='pes')
print('Hessian:')
print(hessian)


# Create a surrogate ParallelLineSearch object
surrogate = generate_surrogate(
    path=base_dir + 'surrogate',
    fname='surrogate.p',  # try to load from disk
    structure=p_relax,
    hessian=hessian,
    pes_func=pes,
    pes_args={'c': c_srg, 'd': d_srg},
    mode='pes',
    window_frac=0.5,  # maximum displacement relative to Lambda of each direction
    M=25,
)

# Optimize the line-search to tolerances
if not surrogate.optimized:
    surrogate.optimize(
        epsilon_p=[0.01, 0.02, 0.03],  # parameter tolerances
        fit_kind='pf3',
        # (initial) maximum resampled noise relative to the maximum window
        noise_frac=0.01,
        M=7,
        N=500,  # use as many points for correlated resampling of the error
    )
    surrogate.write_to_disk('surrogate.p')
# end if

# Define alternative PES
c_alt = 1.0
d_alt = 0.3

# Compute reference minimum


def pes_min_alt(params):
    return pes(ParameterSet(params), c=c_alt, d=d_alt)[0]


# end def
res = minimize(pes_min_alt, p_init.params)
p_relax = ParameterSet(res.x)
print('Minimum-energy parameters (alternative):')
print(p_relax.params)

# Run line-search iteration with the alternative PES
lsi = LineSearchIteration(
    path=base_dir + 'lsi',
    surrogate=surrogate,
    pes_func=pes,
    pes_args={'c': c_alt, 'd': d_alt},
    mode='pes',
)
# Propagate the line-search imax times
imax = 4
for i in range(imax):
    lsi.propagate(i, add_sigma=True)
# end for

linesearch_diagnostics(lsi)
