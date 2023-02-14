#!/usr/bin/env python3

from numpy import polyfit, polyder, polyval, roots, where, argmin, median, array, isnan, linalg, linspace
from numpy import meshgrid, loadtxt

Bohr = 0.5291772105638411  # A
Ry = 13.605693012183622  # eV
Hartree = 27.211386024367243  # eV


# Important function to resolve the local minimum of a curve
def get_min_params(x_n, y_n, pfn = 3, sgn = 1, guess = 0.0, **kwargs):
    assert pfn > 1, 'pfn must be larger than 1'
    pf = polyfit(x_n, y_n, pfn)
    pfd = polyder(pf)
    r = roots(pfd)
    d = polyval(polyder(pfd), r)
    x_mins  = r[where((r.imag == 0) & (sgn * d > 0))].real  # filter real minima (maxima with sgn < 0)
    if len(x_mins) > 0:
        y_mins = polyval(pf, x_mins)
        imin = argmin(abs(x_mins - guess))  # pick the closest to guess
    else:
        x_mins = [min(x_n), max(x_n)]
        y_mins = polyval(pf, x_mins)
        imin = argmin(sgn * y_mins)  # pick the lowest/highest energy
    #end if
    y0 = y_mins[imin]
    x0 = x_mins[imin]
    return x0, y0, pf
#end def


# Estimate conservative (maximum) uncertainty from a distribution based on a percentile fraction
def get_fraction_error(data, fraction, both = False):
    if fraction < 0.0 or fraction > 0.5:
        raise ValueError('Invalid fraction')
    #end if
    data   = array(data, dtype = float)
    data   = data[~isnan(data)]        # remove nan
    ave    = median(data)
    data   = data[data.argsort()] - ave  # sort and center
    pleft  = abs(data[int((len(data) - 1) * fraction)])
    pright = abs(data[int((len(data) - 1) * (1 - fraction))])
    if both:
        err = [pleft, pright]
    else:
        err = max(pleft, pright)
    #end if
    return ave, err
#end def


def match_to_tol(val1, val2, tol = None):
    """Match the values of two vectors. True if all match, False if not."""
    tol = tol if tol is not None else 1e-10
    assert len(val1) == len(val2), 'lengths of val1 and val2 do not match' + str(val1) + str(val2)
    for v1, v2 in zip(val1.flatten(), val2.flatten()):  # TODO: maybe vectorize?
        if abs(v2 - v1) > tol:
            return False
        #end if
    #end for
    return True
#end def


# Map W to R, given H
def W_to_R(W, H):
    R = (2 * W / H)**0.5
    return R
#end def


# Map R to W, given H
def R_to_W(R, H):
    W = 0.5 * H * R**2
    return W
#end def


# courtesy of Jaron Krogel
# XYp = x**0 y**0, x**0 y**1, x**0 y**2, ...
def bipolynomials(X, Y, nx, ny):
    X = X.flatten()
    Y = Y.flatten()
    Xp = [0 * X + 1.0]
    Yp = [0 * Y + 1.0]
    for n in range(1, nx + 1):
        Xp.append(X**n)
    #end for
    for n in range(1, ny + 1):
        Yp.append(Y**n)
    #end for
    XYp = []
    for Xn in Xp:
        for Yn in Yp:
            XYp.append(Xn * Yn)
        #end for
    #end for
    return XYp
#end def bipolynomials


# courtesy of Jaron Krogel
def bipolyfit(X, Y, Z, nx, ny):
    XYp = bipolynomials(X, Y, nx, ny)
    p, r, rank, s = linalg.lstsq(array(XYp).T, Z.flatten(), rcond = None)
    return p
#end def bipolyfit


# courtesy of Jaron Krogel
def bipolyval(p, X, Y, nx, ny):
    shape = X.shape
    XYp = bipolynomials(X, Y, nx, ny)
    Z = 0 * X.flatten()
    for pn, XYn in zip(p, XYp):
        Z += pn * XYn
    #end for
    Z.shape = shape
    return Z
#end def bipolyval


# courtesy of Jaron Krogel
def bipolymin(p, X, Y, nx, ny, itermax = 6, shrink = 0.1, npoints = 10):
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
    #end for
    return xmin, ymin, zmin
#end def bipolymin


def directorize(path):
    if len(path) > 0 and not path[-1] == '/':
        path += '/'
    #end if
    return path
#end def


def match_values(val1, val2, tol = 1e-8, expect_false = False):
    val1 = array(val1).flatten()
    val2 = array(val2).flatten()
    failed = False
    for v,val in enumerate(abs(val1 - val2)):
        if val > tol:
            if not expect_false:
                print('row {}: {} and {} differ by {} > {}'.format(v, val1[v], val2[v], abs(val), tol))
            #end if
            failed = True
        #end if
    #end for
    return not failed
#end def