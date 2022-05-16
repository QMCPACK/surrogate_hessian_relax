#!/usr/bin/env python3

from numpy import linspace, ceil

default_steps = 10

def init_nexus(**nx_settings):
    from nexus import settings
    if len(settings) == 0:
        settings(**nx_settings)
    #end if
#end def

def relax_structure(
    structure,
    relax_job,
    path = 'relax',
    mode = 'nexus',
    j_id = -1,
    make_consistent = True,
    **kwargs
):
    jobs = relax_job(structure, path)
    if mode == 'nexus':
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
        #end if
        pos_relax = ai.structures[len(ai.structures) - 1].positions
        try:
            axes_relax = ai.structures[len(ai.structures) - 1].axes
        except AttributeError:
            axes_relax = None
        #end try
    else: # TODO
        print('Warning: only Nexus currently implemented')
        return None
    #end if
    structure_relax = structure.copy(pos = pos_relax, axes = axes_relax)
    if make_consistent:
        structure_relax.forward()
        structure_relax.backward()
        pos_diff = structure_relax.pos - pos_relax
        print('Max pos_diff was {}'.format(abs(pos_diff).max()))
    #end if
    return structure_relax
#end def


def compute_phonon_hessian(
    structure,
    phonon_job,
    path = 'phonon',
    fc_file = 'FC.fc',
    kind = 'qe',
    mode = 'nexus',
    **kwargs
    ):
    jobs = phonon_job(structure, path)
    if mode == 'nexus':
        from nexus import run_project
        run_project(jobs)
    else:
        print('Warning: only Nexus currently implemented')
    #end if
    if kind == 'qe':
        from surrogate_tools import load_force_constants_qe
        num_elem = len(structure.pos)
        fname = '{}/{}'.format(path, fc_file)
        hessian_real = load_force_constants_qe(fname, num_elem)
        x_unit = 'B'
        E_unit = 'Ry'
    else:
        print('Warning: only QE phonon calculation currently supported')
        return None
    #end if
    from surrogate_classes import ParameterHessian
    # TODO: set target energy
    hessian = ParameterHessian(structure = structure, hessian_real = hessian_real, x_unit = x_unit, E_unit = E_unit)
    return hessian
#end def


def nexus_pwscf_analyzer(path, suffix = 'scf.in', **kwargs):
    from nexus import PwscfAnalyzer
    ai = PwscfAnalyzer('{}/{}'.format(path, suffix))
    ai.analyze()
    E = ai.E
    Err = 0.0
    return E, Err
#end def


def nexus_qmcpack_analyzer(path, qmc_idx = 1, get_var = False, suffix = '/dmc/dmc.in.xml', **kwargs):
    from nexus import QmcpackAnalyzer
    ai = QmcpackAnalyzer('{}/{}'.format(path, suffix))
    ai.analyze()
    LE = ai.qmc[qmc_idx].scalars.LocalEnergy
    LE2 = ai.qmc[qmc_idx].scalars.LocalEnergy_sq
    E     = LE.mean
    Err   = LE.error
    V     = LE2.mean - E**2
    kappa = LE.kappa
    if get_var:
        return E, Err, V, kappa
    else:
        return E, Err
    #end if
#end def


def generate_surrogate(
    structure,
    hessian,
    surrogate_job,
    mode = 'nexus',
    path = 'surrogate',
    epsilon = None,
    generate = True,
    load = None,
    analyze_func = nexus_pwscf_analyzer,
    **kwargs,
):
    if load is not None:
        from surrogate_classes import load_from_disk
        fname = '{}/{}'.format(path, load)
        surrogate = load_from_disk(fname)
        if surrogate is not None:
            return surrogate
        else:
            print('Failed to load {}, generating from scratch...'.format(fname))
        #end if
    #end if
    from surrogate_classes import TargetParallelLineSearch
    surrogate = TargetParallelLineSearch(
        structure = structure,
        targets = structure.params,
        hessian = hessian,
        job_func = surrogate_job,
        path = path,
        **kwargs)
    if generate:
        jobs = surrogate.generate_jobs()
        if mode == 'nexus':
            from nexus import run_project
            run_project(jobs)
            surrogate.load_results(analyze_func = analyze_func, set_target = True, **kwargs)
        else:
            print('Warning: only Nexus currently implemented')
            return None
        #end if
    #end if
    if epsilon is not None:
        pass
    #end if
    return surrogate
#end def


# TODO: make this more fun
def get_color(l):
    colors = 'rgbmck'
    return colors[l % len(colors)]
#end def


from matplotlib import pyplot as plt
def plot_surrogate_pes(
    surrogate,  # surrogate object
    overlay = True,
    **kwargs,
):
    if overlay:
        f, ax = plt.subplots(tight_layout = True)
        ax.set_title('PES: every line-search')
    #end if
    for l, ls in enumerate(surrogate.ls_list):
        if not overlay:
            f, ax = plt.subplots(tight_layout = True)
            ax.set_title('PES: Line-search #{}'.format(l))
        #end if
        label = 'ls #{}'.format(l)
        plot_one_surrogate_pes(ls, ax = ax, color = get_color(l), **kwargs)  # TODO: make this class method
        if not overlay:
            ax.legend(fontsize = 10)
        #end if
    #end for
    if overlay:
        ax.legend(fontsize = 10)
    #end if
#end def


def plot_one_surrogate_pes(
    tls,
    ax,
    marker = '.',
    color = 'k',
    label = '',
    **kwargs
):
    Lambda = tls.Lambda
    grid = tls.target_grid
    values = tls.target_values
    # TODO: diagnose numerically
    if ax is not None:
        xgrid = linspace(grid.min(), grid.max(), 201)
        ygrid = tls.target_in(xgrid)
        ax.plot(xgrid, tls.target_y0 + 0.5*Lambda*xgrid**2, color = color, linestyle = ':', label = '')
        ax.plot(xgrid, ygrid, marker = 'None', color = color, label = '')
        ax.plot(grid, values, marker = marker, color = color, linestyle = 'None', label = '{} data'.format(label))
        
        ax.set_xlabel('Displacement', fontsize = 10)  # TODO: units
        ax.set_ylabel('Energy difference', fontsize = 10)  # TODO: units
    #end if
#end def


def plot_surrogate_bias(
    surrogate,  # surrogate object
    **kwargs,
):
    for l, ls in enumerate(surrogate.ls_list):
        f, ax = plt.subplots(tight_layout = True)
        ax.set_title('Bias: Line-search #{}'.format(l))
        plot_one_surrogate_bias(ls, ax, label = 'ls{}'.format(l), **kwargs)
    #end for
#end def


def plot_one_surrogate_bias(
    tls,
    ax,
    fit_kind = 'pf3',
    M = 7,
    bias_mix = 0.0,
    xcolor = 'k',
    ycolor = 'r',
    tcolor = 'b',
    R_min = 0.0,
    set_x0 = True,
    **kwargs
):
    grid = tls.target_grid
    values = tls.target_values
    xgrid = linspace(grid.min(), grid.max(), 201)
    ygrid = tls.target_in(xgrid)
    bias_mix = bias_mix * tls.Lambda**0.5
    R = linspace(R_min, 0.99999999*grid.max(), 51)
    if set_x0:
        tls.target_x0 = 0.0
        bias_x, bias_y, bias_tot = tls.compute_bias_of(
             R = R,
             fit_kind = fit_kind,
             M = M,
             bias_mix = bias_mix,
             **kwargs,
        )
        tls.target_x0 = bias_x[0]
    #end if
    bias_x, bias_y, bias_tot = tls.compute_bias_of(
        R = R,
        fit_kind = fit_kind,
        M = M,
        bias_mix = bias_mix,
        **kwargs)
    ax.plot(R, bias_x, color = xcolor, linestyle = ':', label = 'x_bias ({})'.format(fit_kind))
    ax.plot(R, bias_y, color = ycolor, linestyle = '--', label = 'y_bias ({})'.format(fit_kind))
    ax.plot(R, bias_tot, color = tcolor, linestyle = '-.', label = 'tot_bias (mix: {:4f})'.format(bias_mix))
    ax.set_xlabel('Grid extent (R)')
    ax.set_ylabel('Bias')
    ax.legend(fontsize = 10)
#end def


def optimize_surrogate(
    surrogate,
    epsilon = None,
    save = None,
    rewrite = True,
    **kwargs,
):
    if not rewrite and surrogate.optimized:
        print('Did not rewrite already optimized surrogate')
        return
    #end if
    surrogate.optimize(epsilon_p = epsilon, **kwargs)
    if not save is None:
        surrogate.write_to_disk(save)
    #end if
#end def


def generate_linesearch(
    surrogate = None,
    job_func = None,
    mode = 'nexus',
    path = 'linesearch',
    load = True,
    load_only = False,
    shift_params = None,
    **kwargs,
):
    from surrogate_classes import LineSearchIteration
    if not load_only:  # a hacky override to enable importing pre-computed modules
        srg = surrogate.copy()
        if not shift_params is None:
            srg.structure.shift_params(shift_params)
        #end if
        lsi = LineSearchIteration(surrogate = srg, path = path, load = load, job_func = job_func, **kwargs)
    else:
        lsi = LineSearchIteration(path = path, load = True, **kwargs)
    #end if
    return lsi
#end def

def propagate_linesearch(
    lsi,
    mode = 'nexus',
    add_sigma = False,
    write = True,
    i = None,
    **kwargs,
):
    # if already calculated, carry on
    if not i is None:
        if lsi.pls(i = i) is None: 
            if i > 0 and lsi.pls(i = i - 1) is None:
                print('Line-search #{} could not be found'.format(i))
                return
            #end if
        elif lsi.pls(i = i).protected:
            print('Line-search #{} is already done and protected'.format(i))
            return
        #end if
    #end if
    # else run and analyze
    if mode == 'nexus':
        from nexus import run_project
        if lsi.pls().protected:
            lsi.propagate(write = False)
        #end if
        run_project(lsi.generate_jobs())
        lsi.load_results(add_sigma = add_sigma, **kwargs)
        lsi.propagate(write = write)
    else:
        print('Not implemented')
        return
    #end if
#end def


def get_qmc_variance(
    structure,
    job_func,
    path = 'path',
    analyzer_func = nexus_qmcpack_analyzer,
    suffix = '/dmc/dmc.in.xml',
    blocks = 200,
    walkers = 1000,
    **kwargs
):
    from nexus import run_project
    run_project(job_func(structure, path, sigma = None))
    res = nexus_qmcpack_analyzer(path, suffix = suffix, get_var = True)
    variance = res[2]
    steps = dmc_steps(sigma = None)
    kappa = res[1]**2 / res[2] * walkers * blocks * steps
    return variance, kappa
#end def


def get_var_eff(
    structure,
    job_func,
    path = 'path',
    analyzer_func = nexus_qmcpack_analyzer,
    suffix = '/dmc/dmc.in.xml',
    equilibration = 10,
    **kwargs
):
    from nexus import run_project
    run_project(job_func(structure, path, sigma = None))
    E, Err = nexus_qmcpack_analyzer(path, suffix = suffix, equilibration = 10)
    var_eff = default_steps * Err**2
    return var_eff
#end def


def dmc_steps(sigma, var_eff = None, variance = 1.0, blocks = 200, walkers = 1000, kappa = 1.0):
    if sigma is None:
        return default_steps
    #end if
    if var_eff is None:
        steps = kappa * variance / sigma**2 / walkers / blocks
    else:
        steps = var_eff / sigma**2
    #end if
    return max(int(ceil(steps)), 1)
#end def


def surrogate_diagnostics(
    surrogate,
    ax = None,
):
    print('Surrogate diagnostics not yet implemented')
    if not ax is None:
        pass
    #end if
#end def


def linesearch_diagnostics(
    lsi,
    ax = None,
):
    print('Linesearch diagonstics not yet implemented')
#end def
