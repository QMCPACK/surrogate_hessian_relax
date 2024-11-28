Introduction
============

This is the User guide for STALK, a Python implementation of the Surrogate Hessian
Accelerated Parallel Line-search method (see :ref:`theory` for details). 

The aim of the method is robust optimization of a noisy cost-function, with the help of a
surrogate, a well-behaving approximation to the cost-function in question. Let us call them
the Target (T) and the Surrogate (S). 

Typically, Target is something of primary interest but also noisy and costly to evaluate,
such as a physical measurement, or a Monte Carlo simulation. The Target runs are
characterized by a set of parameters {p}, and so the evaluation of the Target cost-function
:math:`X_T({p})` returns a scalar value with a finite stochastic errorbar. The cause of
optimization is to find the particular set of parameters :math:`{p}^*` that minimizes the
cost-function :math:`X_T`.

For instance, we could seek the optimal O-H distance and H-O-H bond angle, two parameters,
to find the energy-minimizing geometry of the water molecule, H_2O, where
:math:`X_T` is the evaluated energy.

Many algorithms exist that use parameter gradients of :math:`X_T` to perform optimization,
however, due to the noisy nature of :math:`X_T` this is problematic. Instead, a robust
approach is the direct sampling of :math:`X_T({p})` in an informed manner, to narrow down
:math:`{p}^*` with high efficiency and high confidence. In this implementation, the informed
manner means a sequence of line-searches along the conjugate directions of the cost-function
Hessian :math:`H_T`. At the same, it means mindful requests for statistical precision during
the evaluations of :math:`X_T`. Both of these concepts are vital for performance and better
laid out in :ref:`theory`.

Remarkably, the information about the Hessian and the statistical outlook is not derived
from :math:`X_T` but from the surrogate cost-function, :math:`X_S`. Typically, the Surrogate
is something relatively cheap to evaluate, such as a deterministic numerical calculation.
Presumably, the Surrogate yields qualitatively similar results than Target, only less
accurate, and hence less relevant. Since the Surrogate is only used for acceleration, it
(ideally) does not affect the quality of the outcome.

This is what the Surrogate Hessian Relax software does: it coordinates the intake of
information from (cheap) :math:`X_S`, to accelerate and perform the sampling of (expensive)
:math:`X_T`. The algorithm is oblivious of the choices of :math:`X_T` and :math:`X_S`,
although it can (but needs not necessarily) operate their evaluation through software
bindings. Most notably, Nexus can be used to configure and manage sophisticated numerical
jobs on any computing environment from own laptop to supercomputers. The mapping of
parameters {p} to something ready to evaluate is also a central responsibility of the user,
as it defines very nature of the optimization problem.

These flexibilities require a moderate amount of configuration by the user. Some monitoring
of :math:`X_S` and :math:`X_T` may also be necessary along the way, because the algorithm
cannot decided when they ill-behave, as may happen with numerical recipes. This guide is
written to help out in using the code and understanding why not everything can be automated.