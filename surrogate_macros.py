#!/usr/bin/env python3

from numpy import linspace

def relax_structure(
    structure,
    relax_job,
    path = 'relax',
    mode = 'nexus',
    j_id = -1,
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


def generate_surrogate(
    structure,
    hessian,
    surrogate_job,
    mode = 'nexus',
    path = 'surrogate',
    epsilon = None,
    generate = True,
    **kwargs,
):
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
            surrogate.load_results(analyze_func = nexus_pwscf_analyzer, set_target = True, **kwargs)
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
        plot_one_surrogate_bias(ls, ax, **kwargs)
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
    **kwargs
):
    grid = tls.target_grid
    values = tls.target_values
    xgrid = linspace(grid.min(), grid.max(), 201)
    ygrid = tls.target_in(xgrid)
    bias_mix = bias_mix * tls.Lambda**0.5

    R = linspace(0.0, 0.99999999*grid.max(), 51)
    bias_x, bias_y, bias_tot = tls.compute_bias_of_R(
         R,
         fit_kind = fit_kind,
         M = M,
         bias_mix = bias_mix,
         **kwargs)
    ax.plot(R, bias_x, color = xcolor, linestyle = ':', label = 'x_bias ({})'.format(fit_kind))
    ax.plot(R, bias_y, color = ycolor, linestyle = '--', label = 'y_bias ({})'.format(fit_kind))
    ax.plot(R, bias_tot, color = tcolor, linestyle = '-', label = 'tot_bias (mix: {:4f})'.format(bias_mix))
    ax.set_xlabel('Grid extent (R)')
    ax.set_ylabel('Bias')
    ax.legend(fontsize = 10)
#end def


def optimize_surrogate(
    surrogate,
    epsilon = None,
    **kwargs,
):
    surrogate.optimize(epsilon = epsilon, **kwargs)
#end def


def generate_linesearch(
    surrogate,
    linesearch_job,
    mode = 'nexus',
    path = 'linesearch',
    *kwargs,
):
    lsi = LineSearchIteration(surrogate = surrogate, path = path, **kwargs)
    return lsi
#end def
