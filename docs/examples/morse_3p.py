#!/usr/bin/env python3

# Morse 3p: line-search example
#   3-parameter problem in the abstract parameter space
#
# This example simulates characteristics of the line-search method using
# a lightweight and artificial Morse PES.
#
# Computing task: Runs on command line

from stalk.lsi import LineSearchIteration
from stalk.params import ParameterSet, PesFunction, ParameterHessian
from stalk.pls import TargetParallelLineSearch

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
pes_surrogate = PesFunction(pes, {'c': 1.0, 'd': 0.0})

# Relax numerically in the absence of noise, wrap the function for numerical optimizer
p_relax = p_init.copy()
p_relax.relax(pes_surrogate)
print('Minimum-energy parameters (surrogate):')
print(p_relax.params)

# Compute the numerical Hessian at the minimum parameters using a finite difference method
hessian = ParameterHessian(structure=p_relax)
hessian.compute_fdiff(mode='pes', pes=pes_surrogate)
print('Hessian:')
print(hessian)


# Create, or try to load from disk, surrogate TargetParallelLineSearch object
srg_file = 'surrogate.p'
surrogate = TargetParallelLineSearch(
    mode='pes',
    path='surrogate',
    load='surrogate/' + srg_file,
    structure=p_relax,
    hessian=hessian,
    pes=pes_surrogate,
    M=25,
    window_frac=0.5
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
    surrogate.write_to_disk(srg_file)
# end if

# Define alternative PES
pes_alt = PesFunction(pes, {'c': 0.9, 'd': 0.3})
p_alt = p_relax.copy()
p_alt.relax(pes_alt)
print('Minimum-energy parameters (alternative):')
print(p_alt.params)


# Run line-search iteration with the alternative PES
lsi = LineSearchIteration(
    path=base_dir + 'lsi',
    surrogate=surrogate,
    pes=pes_alt,
    mode='pes',
)
# Propagate the line-search imax times
imax = 4
for i in range(imax):
    lsi.propagate(i)
# end for
print(lsi)