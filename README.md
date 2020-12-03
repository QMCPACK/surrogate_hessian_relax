# Surrogate Hessian Relax

Python3 tools to analyze and cost-optimize relaxation of atomic structures using parallel energy line-searches in the presence of statistical noise.

Currently requires
 * Nexus

Currently supports
 * Quantum Espresso
 * QMCPACK

==OVERVIEW==

The algorithm finds the potential energy minimum making use of two methods: the Surrogate and the Premium.
The surrogate method is cheap and precise, but potentially lacks in accuracy. 
The Premium method has better accuracy but is more costly to evaluate and subject to statistical noise. 
Typical examples are Density Functional Theory (DFT) and Quantum Monte Carlo (QMC), which will be used in the remainder.

The algoritm has the following steps, which will be detailed in the following
1. Surrogate relaxation
2. Surrogate Hessian
3. Surrogate cost-optimization
4. Premium line-search

TODO: complete the README



==SETUP==

describe typical setup


==TROUBLESHOOTING==

describe what can go wrong
