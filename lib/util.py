#!/usr/bin/env python3
"""Various utility functions and constants commonly needed in line-search workflows
"""

from numpy import polyfit, polyder, polyval, roots, where, argmin, median, array, isnan, linalg, linspace
from numpy import meshgrid
from matplotlib import pyplot as plt

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


Bohr = 0.5291772105638411  # A
Ry = 13.605693012183622  # eV
Hartree = 27.211386024367243  # eV


def get_min_params(x_n, y_n, pfn=3, sgn=1, guess=0.0, **kwargs):
    """Find the minimum point by fitting a curve"""
    assert pfn > 1, 'pfn must be larger than 1'
    pf = polyfit(x_n, y_n, pfn)
    pfd = polyder(pf)
    r = roots(pfd)
    d = polyval(polyder(pfd), r)
    # filter real minima (maxima with sgn < 0)
    x_mins = r[where((r.imag == 0) & (sgn * d > 0))].real
    if len(x_mins) > 0:
        y_mins = polyval(pf, x_mins)
        imin = argmin(abs(x_mins - guess))  # pick the closest to guess
    else:
        x_mins = [min(x_n), max(x_n)]
        y_mins = polyval(pf, x_mins)
        imin = argmin(sgn * y_mins)  # pick the lowest/highest energy
    # end if
    y0 = y_mins[imin]
    x0 = x_mins[imin]
    return x0, y0, pf
# end def


def get_fraction_error(data, fraction, both=False):
    """Estimate uncertainty from a distribution based on a percentile fraction"""
    if fraction < 0.0 or fraction > 0.5:
        raise ValueError('Invalid fraction')
    # end if
    data = array(data, dtype=float)
    data = data[~isnan(data)]        # remove nan
    ave = median(data)
    data = data[data.argsort()] - ave  # sort and center
    pleft = abs(data[int((len(data) - 1) * fraction)])
    pright = abs(data[int((len(data) - 1) * (1 - fraction))])
    if both:
        err = [pleft, pright]
    else:
        err = max(pleft, pright)
    # end if
    return ave, err
# end def


def match_to_tol(val1, val2, tol=1e-8):
    """Match the values of two vectors. True if all match, False if not."""
    val1 = array(val1).flatten()
    val2 = array(val2).flatten()
    for diff in abs(val2 - val1):
        if diff > tol:
            return False
        # end if
    # end for
    return True
# end def


def W_to_R(W, H):
    """Map W to R, given H"""
    R = (2 * W / H)**0.5
    return R
# end def


def R_to_W(R, H):
    """Map R to W, given H"""
    W = 0.5 * H * R**2
    return W
# end def


def bipolynomials(X, Y, nx, ny):
    """Construct a bipolynomial expansion of variables

    XYp = x**0 y**0, x**0 y**1, x**0 y**2, ...
    courtesy of Jaron Krogel"""
    X = X.flatten()
    Y = Y.flatten()
    Xp = [0 * X + 1.0]
    Yp = [0 * Y + 1.0]
    for n in range(1, nx + 1):
        Xp.append(X**n)
    # end for
    for n in range(1, ny + 1):
        Yp.append(Y**n)
    # end for
    XYp = []
    for Xn in Xp:
        for Yn in Yp:
            XYp.append(Xn * Yn)
        # end for
    # end for
    return XYp
# end def bipolynomials


def bipolyfit(X, Y, Z, nx, ny):
    """Fit to a bipolynomial set of variables"""
    XYp = bipolynomials(X, Y, nx, ny)
    p, r, rank, s = linalg.lstsq(array(XYp).T, Z.flatten(), rcond=None)
    return p
# end def bipolyfit


def bipolyval(p, X, Y, nx, ny):
    """Evaluate based on a bipolynomial set of variables"""
    shape = X.shape
    XYp = bipolynomials(X, Y, nx, ny)
    Z = 0 * X.flatten()
    for pn, XYn in zip(p, XYp):
        Z += pn * XYn
    # end for
    Z.shape = shape
    return Z
# end def bipolyval


def bipolymin(p, X, Y, nx, ny, itermax=6, shrink=0.1, npoints=10):
    """Find the minimum of a bipolynomial set of variables"""
    for i in range(itermax):
        Z = bipolyval(p, X, Y, nx, ny)
        X = X.ravel()
        Y = Y.ravel()
        Z = Z.ravel()
        imin = Z.argmin()
        xmin = X[imin]
        ymin = Y[imin]
        zmin = Z[imin]
        dx = shrink * (X.max() - X.min())
        dy = shrink * (Y.max() - Y.min())
        xi = linspace(xmin - dx / 2, xmin + dx / 2, npoints)
        yi = linspace(ymin - dy / 2, ymin + dy / 2, npoints)
        X, Y = meshgrid(xi, yi)
        X = X.T
        Y = Y.T
    # end for
    return xmin, ymin, zmin
# end def bipolymin


def directorize(path):
    """If missing, add '/' to the end of path"""
    if len(path) > 0 and not path[-1] == '/':
        path += '/'
    # end if
    return path
# end def


def get_color(line):
    colors = 'rgbmck'
    return colors[line % len(colors)]
# end def


def get_colors(num=1):
    return [get_color(line) for line in range(num)]
# end def


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
