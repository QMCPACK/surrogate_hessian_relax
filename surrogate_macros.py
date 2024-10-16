#!/usr/bin/env python3
'''A selection of utilities for analysis and interfacing with other codes, etc'''

from matplotlib import pyplot as plt
from numpy import linspace, ceil, isscalar, array, zeros, ones, where, mean
from numpy import loadtxt, savetxt, polyfit
from os import makedirs
from os.path import exists
from surrogate_classes import bipolyfit
from lib.parameters import load_xyz, directorize
from lib.util import get_color

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


default_steps = 10

# Minimal function for writing line-search structures


def write_xyz_noise(structure, path, sigma, **kwargs):
    makedirs(path, exist_ok=True)
    structure.write_xyz('{}/structure.xyz'.format(path))
    if sigma is not None:
        savetxt('{}/noise'.format(path), [sigma])
    # end if
    return []
# end def

# Minimal function for writing line-search structures


def write_xsf_noise(structure, path, sigma, **kwargs):
    makedirs(path, exist_ok=True)
    structure.write_xsf('{}/structure.xsf'.format(path))
    if sigma is not None:
        savetxt('{}/noise'.format(path), [sigma])
    # end if
    return []
# end def


# Minimal function for loading energies from disk
def load_E_err(path, sigma=0.0, suffix='energy.dat'):
    fname = path + '/' + suffix
    if not exists(fname):
        print('skipped {}'.format(fname))
        return None, None
    # end if
    data = loadtxt(fname)
    if isscalar(data):
        return data, sigma
    else:
        return data[0], data[1]
    # end if
# end def


def init_nexus(**nx_settings):
    from nexus import settings
    if len(settings) == 0:
        settings(**nx_settings)
    # end if
# end def


def relax_structure(
    structure,
    relax_job,
    path='relax',
    mode='nexus',
    j_id=-1,
    c_pos=1.0,
    pos_file=None,
    make_consistent=True,
    allow_translate=True,
    **kwargs
):
    # try to load from file
    if pos_file is not None:
        try:
            pos_relax = load_xyz('{}/{}'.format(path, pos_file))
            axes_relax = None
            mode = 'load'
        except OSError:
            print('File {} not found'.format(pos_file))
        # end try
    # end if
    jobs = relax_job(structure, path)
    if mode == 'load':
        pass
    elif mode == 'nexus':
        from nexus import run_project
        run_project(jobs)
        job = jobs[j_id]
        if 'pw.x' in job.app_name:  # TODO: make this more robust
            from nexus import PwscfAnalyzer
            ai = PwscfAnalyzer(job)
            ai.analyze()
        else:
            print('Not implemented')
            return None
        # end if
        pos_relax = ai.structures[len(ai.structures) - 1].positions * c_pos
        try:
            axes_relax = ai.structures[len(ai.structures) - 1].axes
            axes_relax *= structure.params[0]
        except AttributeError:
            axes_relax = None
        # end try
    else:  # TODO
        print('Warning: only Nexus currently implemented')
        return None
    # end if
    structure_relax = structure.copy(pos=pos_relax, axes=axes_relax)
    if make_consistent:
        structure_relax._forward(pos_relax)
        structure_relax._backward(structure_relax.params)
        pos_diff = structure_relax.pos - pos_relax
        if allow_translate:
            pos_diff -= pos_diff.mean(axis=0)
        # end if
        print('Max pos_diff was {}'.format(abs(pos_diff).max()))
    # end if
    return structure_relax
# end def


# Load FC matrix in QE format
def load_force_constants_qe(fname, num_prt, dim=3):
    K = zeros((dim * num_prt, dim * num_prt))
    with open(fname) as f:
        line = f.readline()
        while line:
            line_spl = line.split()
            # stupid way to check for integer?
            if len(line_spl) == 4 and len(line_spl[3]) < 3:
                dim1 = int(line_spl[0])
                dim2 = int(line_spl[1])
                prt1 = int(line_spl[2])
                prt2 = int(line_spl[3])
                line = f.readline().split()  # gamma point is the first line
                i = (prt1 - 1) * dim + dim1 - 1
                j = (prt2 - 1) * dim + dim2 - 1
                K[i, j] = float(line[3])
            # end if
            line = f.readline()
        # end if
    # end with
    return K
# end def


def compute_phonon_hessian(
    structure,
    phonon_job,
    path='phonon',
    fc_file='FC.fc',
    kind='qe',
    mode='nexus',
    **kwargs
):
    jobs = phonon_job(structure, path)
    if mode == 'nexus':
        from nexus import run_project
        run_project(jobs)
    else:
        print('Warning: only Nexus currently implemented')
    # end if
    if kind == 'qe':
        num_elem = len(structure.pos)
        fname = '{}/{}'.format(path, fc_file)
        hessian_real = load_force_constants_qe(fname, num_elem)
        x_unit = 'B'
        E_unit = 'Ry'
    else:
        print('Warning: only QE phonon calculation currently supported')
        return None
    # end if
    from surrogate_classes import ParameterHessian
    # TODO: set target energy
    hessian = ParameterHessian(
        structure=structure, hessian_real=hessian_real, x_unit=x_unit, E_unit=E_unit)
    return hessian
# end def


def compute_fdiff_hessian(
    structure,
    func,  # (job_func|pes_func) depending on the mode
    path='fdiff',
    dp=0.01,
    mode='pwscf',
    skip_jobs=0,  # skip as many jobs before analyzing
    **kwargs,
):
    params = structure.params
    P = len(params)
    dps = array(P * [dp]) if isscalar(dp) else dp

    def shift_params(id_ls, dp_ls):
        dparams = array(P * [0.0])
        string = 'scf'
        for p, dp in zip(id_ls, dp_ls):
            dparams[p] += dp
            string += '_p{}'.format(p)
            if dp > 0:
                string += '+'
            # end if
            string += '{}'.format(dp)
        # end def
        structure_new = structure.copy()
        structure_new.shift_params(dparams)
        if mode == 'pes':
            # return list of the energy, parameter displacements
            return [func(structure_new, sigma=0.0)[0]], dparams
        else:
            return func(structure_new, path=path + string), dparams
        # end if
    # end def

    eqm = shift_params([], [])
    jobs = eqm[0]
    pdiffs = [eqm[1]]

    # construct shifts
    for p0, (param0, dp0) in enumerate(zip(params, dps)):
        job, pdiff = shift_params([p0], [+dp0])
        jobs += job
        pdiffs += [pdiff]
        job, pdiff = shift_params([p0], [-dp0])
        jobs += job
        pdiffs += [pdiff]
        for p1, (param1, dp1) in enumerate(zip(params, dps)):
            if p1 <= p0:
                continue
            # end if
            job, pdiff = shift_params([p0, p1], [+dp0, +dp1])
            jobs += job
            pdiffs += [pdiff]
            job, pdiff = shift_params([p0, p1], [+dp0, -dp1])
            jobs += job
            pdiffs += [pdiff]
            job, pdiff = shift_params([p0, p1], [-dp0, +dp1])
            jobs += job
            pdiffs += [pdiff]
            job, pdiff = shift_params([p0, p1], [-dp0, -dp1])
            jobs += job
            pdiffs += [pdiff]
        # end for
    # end for
    pdiffs = array(pdiffs)

    if mode == 'pes':
        energies = array(jobs)
    else:
        from nexus import run_project
        run_project(jobs)
        jobi = 0
        if mode == 'pwscf':
            Es = []
            for job in jobs:
                if jobi < skip_jobs:
                    jobi += 1
                    continue
                # end if
                jobi = 0
                E, Err = nexus_pwscf_analyzer(
                    path=job.locdir, suffix=job.infile)
                Es.append(E)
            # end for
        elif mode == 'gamess':
            Es = []
            for job in jobs:
                if jobi < skip_jobs:
                    jobi += 1
                    continue
                # end if
                jobi = 0
                E, Err = nexus_gamess_analyzer(
                    path=job.locdir, suffix=job.infile)
                Es.append(E)
            # end for
        # end if
        energies = array(Es)
    # end if

    if P == 1:  # for 1-dimensional problems
        pf = polyfit(pdiffs[:, 0], energies, 2)
        hessian = array([[pf[0]]])
    else:
        hessian = zeros((P, P))
        pfs = [[] for p in range(P)]
        for p0, param0 in enumerate(params):
            for p1, param1 in enumerate(params):
                if p1 <= p0:
                    continue
                # end if
                # filter out the values where other parameters were altered
                ids = ones(len(pdiffs), dtype=bool)
                for p in range(P):
                    if p == p0 or p == p1:
                        continue
                    # end if
                    ids = ids & (abs(pdiffs[:, p]) < 1e-10)
                # end for
                XY = pdiffs[where(ids)]
                E = array(energies)[where(ids)]
                X = XY[:, p0]
                Y = XY[:, p1]
                pf = bipolyfit(X, Y, E, 2, 2)
                hessian[p0, p1] = pf[4]
                hessian[p1, p0] = pf[4]
                pfs[p0].append(2 * pf[6])
                pfs[p1].append(2 * pf[2])
            # end for
        # end for
        for p0 in range(P):
            hessian[p0, p0] = mean(pfs[p0])
        # end for
    # end if
    from surrogate_classes import ParameterHessian
    return ParameterHessian(structure=structure, hessian=hessian)
# end def


def nexus_pwscf_analyzer(path, suffix='scf.in', **kwargs):
    from nexus import PwscfAnalyzer
    ai = PwscfAnalyzer('{}{}'.format(directorize(path), suffix))
    ai.analyze()
    E = ai.E
    Err = 0.0
    return E, Err
# end def


def nexus_pwscf_relax(path, suffix='scf.in', **kwargs):
    from nexus import PwscfAnalyzer
    ai = PwscfAnalyzer('{}{}'.format(directorize(path), suffix))
    ai.analyze()
    pos = ai.structures[len(ai.structures) - 1].positions
    try:
        axes = ai.structures[len(ai.structures) - 1].axes
    except AttributeError:
        axes = None
    # end try
    return pos, axes
# end def


def nexus_gamess_analyzer(path, suffix='gamess.inp', allow_fail=True, **kwargs):
    from nexus import GamessAnalyzer
    fname = '{}{}'.format(directorize(path), suffix)
    ai = GamessAnalyzer(fname)
    ai.analyze()
    if 'energy' not in ai.keys() and allow_fail:
        E = 0.0
    else:
        E = ai.energy.total
    # end if
    Err = 0.0
    return E, Err
# end def


def nexus_qmcpack_analyzer(path, qmc_idx=1, get_var=False, suffix='/dmc/dmc.in.xml', **kwargs):
    from nexus import QmcpackAnalyzer
    ai = QmcpackAnalyzer('{}{}'.format(directorize(path), suffix))
    ai.analyze()
    LE = ai.qmc[qmc_idx].scalars.LocalEnergy
    LE2 = ai.qmc[qmc_idx].scalars.LocalEnergy_sq
    E = LE.mean
    Err = LE.error
    V = LE2.mean - E**2
    kappa = LE.kappa
    if get_var:
        return E, Err, V, kappa
    else:
        return E, Err
    # end if
# end def


def generate_surrogate(
    path='surrogate',
    fname=None,
    **kwargs,
):
    if fname is not None:
        from surrogate_classes import load_from_disk
        fname = '{}{}'.format(directorize(path), fname)
        surrogate = load_from_disk(fname)
        if surrogate is not None:
            return surrogate
        else:
            print('Failed to load {}, generating from scratch...'.format(fname))
        # end if
    # end if
    from surrogate_classes import TargetParallelLineSearch
    surrogate = TargetParallelLineSearch(path=path, **kwargs)
    return surrogate
# end def


def plot_surrogate_pes(
    surrogate,  # surrogate object
    overlay=True,
    **kwargs,
):
    if overlay:
        f, ax = plt.subplots(tight_layout=True)
        ax.set_title('PES: every line-search')
    # end if
    for li, ls in enumerate(surrogate.ls_list):
        if not overlay:
            f, ax = plt.subplots(tight_layout=True)
            ax.set_title('PES: Line-search #{}'.format(li))
        # end if
        # label = 'ls #{}'.format(li)
        plot_one_surrogate_pes(ls, ax=ax, color=get_color(
            li), **kwargs)  # TODO: make this class method
        if not overlay:
            ax.legend(fontsize=10)
        # end if
    # end for
    if overlay:
        ax.legend(fontsize=10)
    # end if
# end def


def plot_one_surrogate_pes(
    tls,
    ax,
    marker='.',
    color='k',
    label='',
    **kwargs
):
    Lambda = tls.Lambda
    grid = tls.target_grid
    values = tls.target_values
    # TODO: diagnose numerically
    if ax is not None:
        xgrid = linspace(grid.min(), grid.max(), 201)
        ygrid = tls.target_in(xgrid)
        ax.plot(xgrid, tls.target_y0 + 0.5 * Lambda * xgrid**2, color=color, linestyle=':', label='')
        ax.plot(xgrid, ygrid, marker='None', color=color, label='')
        ax.plot(grid, values, marker=marker, color=color,
                linestyle='None', label='{} data'.format(label))

        ax.set_xlabel('Displacement', fontsize=10)  # TODO: units
        ax.set_ylabel('Energy difference', fontsize=10)  # TODO: units
    # end if
# end def


def plot_surrogate_bias(
    surrogate,  # surrogate object
    **kwargs,
):
    for li, ls in enumerate(surrogate.ls_list):
        f, ax = plt.subplots(tight_layout=True)
        ax.set_title('Bias: Line-search #{}'.format(li))
        plot_one_surrogate_bias(ls, ax, label='ls{}'.format(li), **kwargs)
    # end for
# end def


def plot_one_surrogate_bias(
    tls,
    ax,
    fit_kind='pf3',
    M=7,
    bias_mix=0.0,
    xcolor='k',
    ycolor='r',
    tcolor='b',
    R_min=0.0,
    set_x0=True,
    **kwargs
):
    # grid = tls.target_grid
    # values = tls.target_values
    # xgrid = linspace(grid.min(), grid.max(), 201)
    # ygrid = tls.target_in(xgrid)
    bias_mix = bias_mix * tls.Lambda**0.5
    R = linspace(R_min, 0.99999999 * tls.R_max, 51)
    if set_x0:
        tls.target_x0 = 0.0
        bias_x, bias_y, bias_tot = tls.compute_bias_of(
            R=R,
            fit_kind=fit_kind,
            M=M,
            bias_mix=bias_mix,
            **kwargs,
        )
        tls.target_x0 = bias_x[0]
    # end if
    bias_x, bias_y, bias_tot = tls.compute_bias_of(
        R=R,
        fit_kind=fit_kind,
        M=M,
        bias_mix=bias_mix,
        **kwargs)
    ax.plot(R, bias_x, color=xcolor, linestyle=':',
            label='x_bias ({})'.format(fit_kind))
    ax.plot(R, bias_y, color=ycolor, linestyle='--',
            label='y_bias ({})'.format(fit_kind))
    ax.plot(R, bias_tot, color=tcolor, linestyle='-.',
            label='tot_bias (mix: {:4f})'.format(bias_mix))
    ax.set_xlabel('Grid extent (R)')
    ax.set_ylabel('Bias')
    ax.legend(fontsize=10)
# end def


def optimize_surrogate(
    surrogate,
    epsilon=None,
    save=None,
    rewrite=True,
    **kwargs,
):
    if not rewrite and surrogate.optimized:
        print('Did not rewrite already optimized surrogate')
        return
    # end if
    surrogate.optimize(epsilon_p=epsilon, **kwargs)
    if save is not None:
        surrogate.write_to_disk(save)
    # end if
# end def


def generate_linesearch(
    surrogate=None,
    path='linesearch/',
    load=True,
    load_only=False,
    shift_params=None,
    mode='nexus',
    **kwargs,
):
    from surrogate_classes import LineSearchIteration
    if not load_only:  # a hacky override to enable importing pre-computed modules
        srg = surrogate.copy()
        if shift_params is not None:
            srg.structure.shift_params(shift_params)
        # end if
        lsi = LineSearchIteration(
            surrogate=srg, path=path, load=load, **kwargs)
    else:
        lsi = LineSearchIteration(path=path, load=True, **kwargs)
    # end if
    return lsi
# end def


def propagate_linesearch(
    lsi,
    mode='nexus',
    add_sigma=False,
    write=True,
    i=None,
    **kwargs,
):
    # if already calculated, carry on
    if i is not None:
        if lsi.pls(i=i) is None:
            if i > 0 and lsi.pls(i=i - 1) is None:
                print('Line-search #{} could not be found'.format(i))
                return
            # end if
        elif lsi.pls(i=i).status.protected:
            print('Line-search #{} is already done and protected'.format(i))
            return
        # else:  #FIXME: treat i = i case properly
        # end if
    # end if
    # else run and analyze
    if mode == 'pes':
        lsi.propagate(mode=mode, **kwargs)
    elif mode == 'nexus':
        from nexus import run_project
        if lsi.pls().status.protected:
            lsi.propagate(write=False)
        # end if
        run_project(lsi.generate_jobs(**kwargs))
        lsi.load_results(add_sigma=add_sigma, **kwargs)
        lsi.propagate(write=write)
    else:
        print('Propagation mode {} not implemented'.format(mode))
        return
    # end if
# end def


def get_qmc_variance(
    structure,
    job_func,
    path='path',
    analyzer_func=nexus_qmcpack_analyzer,
    suffix='/dmc/dmc.in.xml',
    blocks=200,
    walkers=1000,
    analyzer_args={},
    **kwargs
):
    from nexus import run_project
    run_project(job_func(structure, path, sigma=None))
    res = nexus_qmcpack_analyzer(
        path, suffix=suffix, get_var=True, **analyzer_args)
    variance = res[2]
    steps = dmc_steps(sigma=None)
    kappa = res[1]**2 / res[2] * walkers * blocks * steps
    return variance, kappa
# end def


def get_var_eff(
    structure,
    job_func,
    path='path',
    analyzer_func=nexus_qmcpack_analyzer,
    suffix='/dmc/dmc.in.xml',
    equilibration=10,
    analyzer_args={},
    **kwargs
):
    from nexus import run_project
    run_project(job_func(structure.copy(), path, sigma=None))
    E, Err = nexus_qmcpack_analyzer(
        path, suffix=suffix, equilibration=10, **analyzer_args)
    var_eff = default_steps * Err**2
    return var_eff
# end def


def dmc_steps(sigma, var_eff=None, variance=1.0, blocks=200, walkers=1000, kappa=1.0):
    if sigma is None:
        return default_steps
    # end if
    if var_eff is None:
        steps = kappa * variance / sigma**2 / walkers / blocks
    else:
        steps = var_eff / sigma**2
    # end if
    return max(int(ceil(steps)), 1)
# end def


def surrogate_diagnostics(
    surrogate,
    show_plot=True,
):
    print('Surrogate diagnostics:')
    for li, ls in enumerate(surrogate.ls_list):
        print('  line-search #{}:'.format(ls.d))
        print('    Lambda:     {}'.format(ls.Lambda))
        print('    fit_kind:   {}'.format(ls.fit_kind))
        print('    M:          {}'.format(ls.M))
        print('    W:          {}'.format(ls.W))
        print('    R:          {}'.format(ls.R))
        print('    epsilon:    {}'.format(ls.epsilon))
        print('    W_opt:      {}'.format(ls.W_opt))
        print('    sigma_opt:  {}'.format(ls.sigma_opt))
        print('    bias:       {}'.format(surrogate.get_biases_d()[li]))
    # end for

    print('Error scan:')
    for p, param in enumerate(surrogate.structure.params):
        print('  parameter #{}:'.format(p))
        print('    value:      {}'.format(param))
        print('    Lambda:     {}'.format(surrogate.hessian.hessian[p, p]))
        if surrogate.epsilon_p is not None:
            print('    epsilon:    {}'.format(surrogate.epsilon_p[p]))
        # end if
        print('    errors:     {}'.format(surrogate.error_p[p]))
        print('    bias:       {}'.format(surrogate.get_biases_p()[p]))
    # end for

    if show_plot:
        pass  # WIP
    # end if
# end def


def linesearch_diagnostics(
    lsi,
    show_plot=True,
):
    print('Linesearch diagnostics:\n')

    print('Parameter convergence:')
    params, params_err = lsi.get_params()
    for i, (param, param_err) in enumerate(zip(params, params_err)):
        fmt = 'pls{}  '
        pedata = []
        for p, pe in zip(param, param_err):
            fmt += ' {:<6f} +/- {:<6f}'
            pedata += [p, pe]
        # end for
        print(fmt.format(i, *tuple(pedata)))
    # end for

    if show_plot:
        lsi.plot_convergence()
    # end if
# end def
