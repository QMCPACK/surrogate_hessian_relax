from numpy import array
from matplotlib import pyplot as plt


def plot_parameter_convergence(
    pls_list,
    ax=None,
    separate=True,
    P_list=None,
    colors=None,
    markers=None,
    marker='x',
    linestyle=':',
    uplims=False,
    lolims=False,
    labels=None,
    targets=None,
    **kwargs,
):
    pls0 = pls_list[0]
    if P_list is None:
        P_list = range(len(pls0.structure.params))
    # end if
    P = len(P_list)
    if ax is None and not separate:
        f, ax = plt.subplots()
    # end if
    targets = array(targets) if targets is not None else pls0.structure.params
    markers = markers if markers is not None else P * [marker]
    colors = colors if colors is not None else get_colors(P)
    labels = labels if labels is not None else ['p' + str(p) for p in P_list]

    # init values
    x_grids = []
    P_vals = []
    P_errs = []
    for p in P_list:
        P_vals.append([pls0.structure.params[p] - targets[p]])
        P_errs.append([0.0])
        x_grids.append([0.0])
    # end for
    # line search params
    for li, pls in enumerate(pls_list):
        if pls.status.analyzed:
            for p in P_list:
                x_grids[p].append(li + 1)
                P_vals[p].append(pls.structure_next.params[p] - targets[p])
                P_errs[p].append(pls.structure_next.params_err[p])
            # end for
        # end if
    # end for
    # plot
    for p in P_list:
        P_val = P_vals[p]
        P_err = P_errs[p]
        xgrid = x_grids[p]
        if p in P_list:
            if separate:
                f, ax = plt.subplots()
                ax.set_xticks(xgrid)
                ax.plot([0, len(pls_list)], [0, 0], 'k-')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Parameter value')
                ax.set_title('Parameters vs iteration')
            # end if
            h, c, f = ax.errorbar(
                xgrid,
                P_val,
                P_err,
                color=colors[p],
                marker=markers[p],
                linestyle=linestyle,
                label=labels[p],
                uplims=uplims,
                lolims=lolims,
                **kwargs,
            )
            if uplims or lolims:
                c[0].set_marker('_')
                c[1].set_marker('_')
            # end if
        # end if
    # end for
    if not separate:
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Parameter value')
        ax.set_title('Parameters vs iteration')
        ax.set_xticks(xgrid)
        ax.plot([0, len(pls_list)], [0, 0], 'k-')
    # end if
# end def


def plot_energy_convergence(
    pls_list,
    ax=None,
    color='tab:blue',
    marker='x',
    linestyle=':',
    uplims=False,
    lolims=False,
    **kwargs,
):
    if ax is None:
        f, ax = plt.subplots()
    # end if

    # init values
    E_vals, E_errs, x_grid = [], [], []
    for p, pls in enumerate(pls_list):
        if pls.structure.value is not None:
            x_grid.append(p)
            E_vals.append(pls.structure.value)
            E_errs.append(pls.structure.error)
        # end if
    # end for

    # plot
    h, c, f = ax.errorbar(
        x_grid,
        E_vals,
        E_errs,
        marker=marker,
        color=color,
        linestyle=linestyle,
        uplims=uplims,
        lolims=lolims,
        **kwargs,
    )
    if uplims or lolims:
        c[0].set_marker('_')
        c[1].set_marker('_')
    # end if
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy value')
    ax.set_title('Energy vs iteration')
    ax.set_xticks(x_grid)
# end def


def plot_bundled_convergence(
    pls_list,
    P_list=None,
    colors=None,
    markers=None,
    labels=None,
    targets=None,
    **kwargs
):
    f, [ax0, ax1] = plt.subplots(2, 1, sharex=True)
    plot_parameter_convergence(
        pls_list,
        P_list=P_list,
        separate=False,
        ax=ax0,
        targets=targets,
        colors=colors,
        markers=markers,
        labels=labels,
        **kwargs)
    plot_energy_convergence(
        pls_list,
        ax=ax1,
        **kwargs
    )
    f.align_ylabels()
# end def


def get_color(line):
    colors = 'rgbmck'
    return colors[line % len(colors)]
# end def


def get_colors(num=1):
    return [get_color(line) for line in range(num)]
# end def
