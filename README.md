# Surrogate Hessian Relax

Python3 tools to analyze and cost-optimize relaxation of atomic structures using parallel energy line-searches in the presence of statistical noise.


## OVERVIEW

The Surrogate Hessian Relax algorithm is intended for relaxing by energy minimization atomic structures with reduced set of parameters. 
The energy minimization is driven by two methods, which are here called the Surrogate and the Main method.
The Surrogate method is cheap and smooth but has lower accuracy, such as Density Functional Theory (DFT).
The Main method has higher accuracy but is more costly to evaluate and subject to statistical noise, such as Quantum Monte Carlo (QMC).

The potential energy surface (PES) and Hessian of the Surrogate method are used to accelerate line-searches in the PES of the Main method, where the Hessian is unavailable.
The number of line-searches is equal to the number of structural parameters, which can be reduced based on symmetries, or simply by choice.
For instance, the atomic structure of benzene molecule has 36 degrees of freedom, which can be reduced to just two bond-length parameters (see Examples).
The Hessian reconstruction of the PES along these reduced parameters can be decoupled, allowing parallel, independent line-searches along special directions.
The equilibrium parameters can be found from any displaced position in a single iteration, comprising simultaenous line-search along all directions, if the displacement is within region where the PES is quadratic.
Even if the Surrogate Hessian differs slightly from the PES, the line-search typically converges in few iteration.

The Surrogate PES can also be used to map and quantify sources of error in the line-search, which are not related to the iteration process.
The line-searches are resolved by finding minima of polynomial fits, which is subject errors due to systematic bias (anharmonicity), statistical noise, and statistical bias. 
The algorithm contains tools to map and control these effect, making it possible to target systematic and stochastic accuracy, and optimize cost of running the Main method. 

Despite the focus on atomic structures, the method, including this implementation, can be generalized for any cost-minimization problem, where the Main method is subject to statistical noise, and a smooth, noiseless Surrogate method is available.




## INSTALLATION


This section is a minimal guide for installing and running the Surrogate Hessian Relaxation guide, and using templates to speed the overall setup.
The theoretical background and better details of each step are given later in this documentation.

### Software requirements

To install, make sure to meet the following minimum requirements for Python.

* Python3
  * Numpy
  * Matplotlib
  * Nexus

Furthermore, the libraries of Nexus and the root directory of Surrogate Hessian Relax, containing `surrogate_tools.py`, `surrogate_relax.py`, and `surrogate_error_scan.py` need to be in the Python environment. 
For instance, include these locations in `PYTHONPATH` variable.

To make best use of the algorithm, it is preferable to have at least one Surrogate and one Main method.

* Surrogate methods available
  * Quantum Espresso (https://https://www.quantum-espresso.org/)

* Main methods available
  * QMCPACK (https://qmcpack.org)


### Using Nexus

Currently the framework relies on Nexus, a python-based workflow tool. 
Nexus is interfaced with various software for electronic structure calculations and computing environments, including local workstations and supercomputers with job managers.
It operates on minimal input from the user to set up, launch and analyze a specific calculations, such as computing the energy of an atomic structure with a given method.

It will be left to the user to provide python functions that return the Surrogate method and the Main method as Nexus jobs. This can be done in any way preferred.
The template file for nexus job definitions is `parameters.py` in the project directory.

New users are referred to the provided examples and the Nexus online documentation:  https://nexus-workflows.readthedocs.io/en/latest/


### Using templates

A full relaxation run includes the following files and directories:
```
./control/
  -> error_scan/
  -> main/
  -> parameters.py
  -> run_relax.py
  -> run_phonon.py
  -> run_error_scan.py
  -> run_ls.py
./relax/
./phonon/
  -> ph.in
  -> q2r.in
./pseudos/
```
The main directory is for relaxing a structure with specific parameterization and line-search options is `./control`, which can be rename accordingly.
It exists to differentiate from other possible parameterizations. 

The `./relax/` directory contains input files and data for Surrogate relaxation, which is created, executed and analyzed by running `control/run_relax.py` template. 
Similarly, the `./phonon/` directory contains input files and data for Surrogate phonon calculation, which is executed manually but analyzed with `control/run_phonon.py`. 
Template input files `phonon/ph.in` and `phonon/q2r.in` are provided but may need adjustment by the user.
The directories `./relax/`, `./phonon/`, and pseudopotential location `./pseudos` are located outside of `./control` to emphasize that they are generally common between different controls directories.

The subdirectory `control/error_scan` contains input file and data for error scanning runs that are created and analyzed by running `control/run_error_scan.py` template. The template file needs to be edited according to the users needs.

The subdirectory `control/main` contains input file and data for Main line-search relaxation runs that are created and analyzed by running `control/run_ls.py` template. 

The above comprises the minimal workflow to achieve line-search, where the templates contain dependencies that are met in the order presented:
1. `run_relax.py`
1. `run_phonon.py`
1. `run_error_scan_py`
1. `run_ls.py`
Therefore, with proper settings that are defined in `parameters.py` and some of the templates, it is possible for the user to only execute the last step, and the cascade or prerequisite steps is executed accordingly.
In practice it is likely useful to diagnose the results of each step before moving on.

It is possible to define multiple cascades involving, e.g. different error scans and multiple variations of the Main method. Then, the script filenames, dependencies and target directories must be changed accordingly.
For instance, a useful and cheap sanity test is to simulate the line-search relaxation by running using the Surrogate method, starting from a displaced position and adding artificial noise.

Generally, it is not mandatory to use the above layout; it is just the one assumed in the provided templates and examples. One is welcome to, e.g., contain everything in one directory or include the whole workflow in a single python script.




## SETUP

The algoritm has the following steps, which will be briefly described later in this documentation
1. Parameterization
1. Surrogate relaxation
1. Surrogate Hessian
1. Surrogate cost-optimization
1. Main line-search


### Parameterization

The relaxation is done according to a reduced set of physical parameters.
Typical parameters include bond lengths, bond angles, and cell parameters.
In principle there are infinite ways of parameterizing an atomic structure, although typically an appropriate parameterization arises naturally or is already decided.

The choice of parameterization affects the Hessian, and thus, optimal directions and line-search performance in some way, which may not be obvious or too important. 
Importantly, the set of parameters must be linearly independent, to avoid null modes from the Hessian, and they must span the degrees of freedom consistent with the user's desires.
Furthermore, the parameters must be scalar-valued (set of parameters is 1-dimensional array) and analytically well-behaved (PES is smooth and Hessian exists).
Otherwise, there are no limits to choosing the parameterization.

The price of this freedom is that the parameters mappings must be figured out and implemented by the user. 
This can be the most creative part of using the algorithm, which boils down to writing explicit array mappings to produce parameters from positions, and vice versa.

#### Forward mapping

The general template of the forward mapping takes position array `pos` and produces an array of parameters `params`. The initial cell `cell_init` is given earlier in the file.

```
def pos_to_params(pos, cell=cell_init, **kwargs):
    # make measurements from pos and cell to populate numpy array params
    return params
#end def
```

#### Backward mapping

The general template of the backward mapping takes parameter array `params` and produces an array of positions `pos`. If the cell is involved, use
```
def params_to_pos(params, give_cell=True, **kwargs):
    # make measurements from pos and cell to populate numpy array params
    if give_cell:
        return pos
    else:
        return pos,cell
    return 
#end def
```
and if not,
```
def params_to_pos(params, **kwargs):
    # make measurements from pos and cell to populate numpy array params
    return pos
#end def
```


### Surrogate relaxation

TODO

### Surrogate Hessian

TODO

### Surrogate cost-optimization

TODO

### Main line-search

TODO


## EXAMPLES


## TROUBLESHOOTING

TODO


## ACKNOWLEDGEMENTS


