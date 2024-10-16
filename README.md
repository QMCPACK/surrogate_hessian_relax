# Surrogate Hessian Relax

Surrogate Hessian Relax is a Python implementation of The Surrogate Hessian
Accelerated Parallel Line-search. The method is inteded for optimizing and
performing energy minimization of atomic structures in the presence of
statistical noise.

NOTE: the implementation is currently in an early stage of development.


## ORIGINAL WORK

The method has been published in The Journal of Chemical Physics as 
[Surrogate Hessian Accelerated Structural Optimization for Stochastic Electronic Structure Theories](https://doi.org/10.1063/5.0079046)

Upon publishing results based on the method, we kindly ask you to cite

> Juha Tiihonen, Paul R. C. Kent, and Jaron T. Krogel \
The Journal of Chemical Physics \
156, 054104 (2022)


## OVERVIEW

The Surrogate Hessian Accelerated Parallel Line-search is an
algorithm for relaxing atomic structures by energy minimization, in the presence
of statistical noise. It is based on conjugate gradient descent (or parallel
line-search), and using a surrogate method to characterize the potential energy
surface (PES). 

For more details on the scope and the theoretical background, we refer to 
the [Original work](#original-work).

The energy minimization involves two electronic structure theories or methods, which will
be called the Surrogate and the Stochastic method. The Surrogate method is cheap and
smooth but has limited accuracy, such as Density Functional Theory (DFT). The
Stochastic method has higher accuracy but is more costly to evaluate and subject to
statistical noise, such as Quantum Monte Carlo (QMC).

An overview of the algorithm is presented below in a simplified graph:

![Overview](docs/overview.png)

The structural relaxation is indeed done in five steps:
1. Surrogate: Relaxation
1. Surrogate: Parameter Hessian
1. Surrogate: Line-search optimization
1. Stochastic: Line-search
1. Stochastic: Finding new minimum

Background and instructions for setting up each of the steps is given in the
main [Documentation](docs/) (work-in-progress).


## INSTALLATION

To install, make sure to meet the following minimum requirements for Python.

* Python (3.12.5)
  * Numpy (2.0.1)
  * Scipy (1.14.1)
  * Dill (0.3.9)
  * Matplotlib (3.9.2)
  * Nexus (https://qmcpack.org)

The libraries of Nexus and the root directory of Surrogate Hessian Relax,
containing `surrogate_tools.py`, `surrogate_relax.py`, and
`surrogate_error_scan.py` need to be in the Python environment. 
For instance, include these locations in the `PYTHONPATH` variable.

The support of various Surrogate and Stochastic methods is currently implemented
through Nexus. The following list contains the software and methods with which the
toolbox has been used and which are covered in the present set of examples:

* Surrogate methods:
  * Quantum Espresso (https://https://www.quantum-espresso.org/)

* Stochastic methods:
  * QMCPACK (https://qmcpack.org)

However, it is possible to use Surrogate Hessian Relax without
direct Nexus support (will be documented).


## LITERATURE

List of scientific works demonstrating the The Surrogate Hessian Accelerated Parallel
Line-search method:
* [J. Chem. Phys, 156, 054104 (2022)](https://doi.org/10.1063/5.0079046) Surrogate Hessian Accelerated Structural Optimization for Stochastic Electronic Structure Theories
* [J. Chem. Phys. 156, 014707 (2022)](https://aip.scitation.org/doi/10.1063/5.0074848) A combined first principles study of the structural, magnetic, and phonon properties of monolayer CrI3
* [Phys. Rev. Materials 5, 024002 (2021)](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.5.024002) Optimized structure and electronic band gap of monolayer GeSe from quantum Monte Carlo methods


## SUPPORT

While the theoretical method has been successfully demonstrated, the software
implementation is currently under major development.

On one hand this means that unexperiences users should expect some 
troubleshooting before successful completion of a line-search project. On the
other hand, the code design and conventions will be subject to change until 
the first stable version is published.

Support can be inquired by [contacting the authors](mailto:juha.tiihonen@tuni.fi).


## ACKNOWLEDGEMENTS

The authors of this method are Juha Tiihonen, Paul R. C. Kent and Jaron T.
Krogel, working in the Center for Predictive Simulation of Functional Materials
(https://cpsfm.ornl.gov/)

This work has been authored in part by UT-Battelle, LLC, under contract
DE-AC05-00OR22725 with the US Department of Energy (DOE).
