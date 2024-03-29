#!/usr/bin/env python3

# Functions and methods related to surrogate error scan

from numpy import array, meshgrid, polyfit, polyval, argmin, linspace, random
from numpy import amax, argmax, isnan, var, amin, isscalar, argsort
from scipy.interpolate import interp1d, pchip_interpolate
from matplotlib import pyplot as plt
from functools import partial
from scipy.optimize import broyden1
from scipy.ndimage import zoom

from iterationdata import IterationData
from surrogate_tools import W_to_R, R_to_W, get_min_params, get_fraction_error, model_statistical_bias


# Function to scan for maximum fitting biases in all directions to approximately stay within maximum tolerances epsilon.
# The W_max will guide in choosing grids for the surrogate PES runs with the caveat that larger errors than epsilon may be left outside of the grid.
# Therefore maximum realistically tolerated epsilon should be used to load_W_max
#   epsilon can be a scalar or an array adjusted for each direction
def load_W_max(
    data,
    epsilon,
    pfn,
    pts,
    W_min   = 1.0e-3,
    verbose = False,
    energy_mix = 0.0,
):
    if isscalar(epsilon):
        epsilon = data.D * [epsilon]
    #end if
    if isscalar(W_min):
        W_min = data.D * [W_min]
    #end if
    Wmaxs = []
    y0    = data.PES[0].min()
    for d in range(data.D):
        x_n   = data.Dshifts[d]
        Rmax  = min(max(x_n), -min(x_n))
        y_n   = data.PES[d]
        H     = data.Lambda[d]
        W_eff = R_to_W(Rmax, H)
        Ws    = linspace(min(W_min[d], 0.01 * W_eff), 0.99999 * W_eff, 51)
        xBs   = []
        yBs   = []
        for W in Ws:
            R        = W_to_R(W, H)
            x_r      = linspace(-R, R, pts)
            x_r, y_r = interpolate_grid(x_n, y_n, x_r)
            y, x, p  = get_min_params(x_r, y_r, pfn)
            x_B      = x
            y_B      = (y - y0)
            xBs.append(x_B)
            yBs.append(y_B)
        #end for
        # try to correct numerical biases due to bad relaxation by subtracting bias near low-W limit
        xB_in = interp1d(Ws, xBs - xBs[0], kind = 'cubic')
        yB_in = interp1d(Ws, yBs - yBs[0], kind = 'cubic')
        if verbose:
            print('W        bias      energy bias       mix')
            for W in Ws:
                print('{:10f} {:10f} {:10f} {:10f}'.format(W, xB_in(W), yB_in(W), xB_in(W) + energy_mix * yB_in(W)))
            #end for
        #end if
        Wmax = 0.0
        for W in Ws:
            # break if bias gets too large for any parameter
            if any(abs(xB_in(W) * data.U[d, :]) - epsilon > 0):
                Wmax = W
                break
            #end if
        #end for
        if Wmax == 0:
            Wmax = 0.99999 * W_eff
            Bmax = round(max(abs(xB_in(Wmax) * data.U[d, :]) / epsilon) * 100, 0)
            print('Warning: Bias in direction {} is only {}% at W_max {}'.format(d, Bmax, Wmax))
            print(d, Wmax, abs(xB_in(Wmax) * data.U[d, :]) - epsilon)
        #end if
        Wmaxs.append(Wmax)
    #end for
    return Wmaxs
#end def


# Function to scan line-search in all directions using correlated resampling over the surrogate PES data
# Operates on regular grids based on energy window W and input noise sigma that can be adjusted in the input
# It is somewhat critical to set the W-sigma grid properly (trial & error before this gets heuristically automated):
#   W grid should span values from small (to assess zero bias) to as large as relevant (load_W_max helps here)
#      if W_max is too low (compared to sigma_max), the grid is poorly focused
#   sigma grid should span values from small (zero noise) to as large as relevant (this can be hard to guess)
#      if sigma_max is too low, it underestimates the optimal noise (contour caps at the sigma_max limit)
#      if sigma_max is too high, most error values on the grid are saturated (poorly focused) and the modeling sigma_opt(epsilon) is unreliable or impossible
def scan_error_data(
    data,
    pfn,
    pts,
    sigma_max = None,
    W_max     = None,
    W_min     = 1.0e-3,
    W_num     = 11,
    sigma_num = 11,
    generate  = 1000,
    relative  = True,  # obsolete, here for compatibility
    fraction  = None,
):
    if fraction is None:
        fraction = data.fraction
    #end if
    if isscalar(W_max):
        W_max = data.D * [W_max]
    elif W_max is None:
        W_max = data.windows
    #end if
    if isscalar(sigma_max) or sigma_max is None:  # if None, increase incrementally
        sigma_max = data.D * [sigma_max]
    #end if

    Xs  = []
    Ys  = []
    Es  = []
    Bs  = []
    Gs  = []
    for d in range(data.P):
        x_n      = data.Dshifts[d]
        y_n      = data.PES[d]
        H        = data.Lambda[d]
        X, Y, E, B, G = scan_linesearch_error(
            x_n,
            y_n,
            H,
            pts       = pts,
            pfn       = pfn,
            W_num     = W_num,
            W_max     = W_max[d],
            W_min     = W_min,
            sigma_num = sigma_num,
            sigma_max = sigma_max[d],
            sigma_min = 0.0,
            generate  = generate,
            fraction  = fraction,
        )
        Xs.append(X)
        Ys.append(Y)
        Es.append(E)
        Bs.append(B)
        Gs.append(G)
    #end for
    data.Xs  = Xs
    data.Ys  = Ys
    data.Es  = Es
    data.Bs  = Bs
    data.Gs  = Gs
    data.Bcs = None
    data.pts = pts
    data.pfn = pfn
    data.fraction = fraction
#end def


# takes a set of points, hessian, parameters to define W and sigma grid
#   x_n
#   y_n
#   H
# returns W x sigma grid and total errors on it
#   X, Y
#   errors
#   systematic bias
def scan_linesearch_error(
    x_n,
    y_n,
    H,
    pfn       = 3,
    pts       = 7,
    W_num     = 7,
    W_min     = None,
    W_max     = None,
    sigma_num = 7,
    sigma_min = 0.0,
    sigma_max = None,
    generate  = 1000,
    fraction  = 0.159,
):

    Rmax  = min(max(x_n), -min(x_n))
    W_eff = R_to_W(Rmax, H)

    if sigma_max is None:
        sigma_max = W_eff / 16  # max fluctuation set to 1/16 of effective W
    #end if
    sigmas = linspace(sigma_min, sigma_max, sigma_num)
    if W_max is None:
        W_max = W_eff
    #end if
    if W_min is None or W_min > W_max:
        W_min = W_max / W_num
    #end if
    Ws   = linspace(W_min, W_max, W_num)
    Gs = random.randn(generate, pts)
    endpts = [min(x_n), max(x_n)]  # evaluate at the end points
    X, Y = meshgrid(Ws, sigmas)

    print('sigma #:   ' + str(sigma_num))
    print('sigma min: ' + str(sigma_min))
    print('sigma max: ' + str(sigma_max))
    print('W #:       ' + str(W_num))
    print('W min:     ' + str(W_min))
    print('W max:     ' + str(W_max))
    print('Using fraction=%f' % fraction)
    Es = []
    Bs = []
    first = True
    for w, W in enumerate(Ws):
        R        = W_to_R(W, H)
        x_r      = linspace(-R, R, pts)
        x_r, y_r = interpolate_grid(x_n, y_n, x_r)
        y, x, p  = get_min_params(x_r, y_r, pfn)
        if first:
            B0    = x  # try to compensate for bias due to relaxation errors
            #y0    = y
            first = False
        #end if
        B        = x  # systematic bias
        #B        = x + (y-y0)/H**0.5
        Bs.append(B)

        E_w = []
        for s, sigma in enumerate(sigmas):
            xdata = []
            for n in range(generate):
                y_min, x_min, pf = get_min_params(x_r, y_r + sigma * Gs[n], pfn, endpts = endpts)
                xdata.append(x_min)
            #end for
            Aave, Aerr = get_fraction_error(array(xdata) - B + B0, fraction = fraction)
            m = model_statistical_bias(p, x_r, sigma)
            #E = Aerr + abs(B-B0) + m # exact systematic bias, model statistical bias instead of Aave
            E = Aerr + abs(B - B0 + m)  # exact systematic bias, model statistical bias instead of Aave
            E_w.append(E)
        #end for
        Es.append(E_w)
    #end for
    Es = array(Es).T
    Bs = array(Bs)

    return X, Y, Es, Bs, Gs
#end def


def calculate_R2(y, y_res):
    return 1 - sum(array(y_res)**2) / var(array(y))
#end def


# Functions to read pre-scanned line-search error grids from an IterationData object and fit a simple regression model the optimal parameters
# The optimal parameter for W and sigma are based on the max-sigma point on an error isocontour on the W-sigma grid
#   It is assumed that
# The fit is based on evaluating the optimum points on an irregular grid of tolerance values (epsilon) and then fitting to assumed scaling models
#   optimal window scales: W_opt(epsilon)**2  ~ epsilon     <- fitted linearly
#   optimal noise scales:  sigma_opt(epsilon) ~ epsilon**2  <- fitted harmonically
# The point of fitting to simple models is fast evaluation (contouring is slow) in later parts, where the optimal points are explored to find the optimal ensemble among all directions
def load_of_epsilon(
    data,
    gridexp   = 4.0,
    show_plot = False,
    get_data  = False,
    epsilons  = None,
    R2thrs    = 0.0
):
    Wfuncs = []
    Sfuncs = []
    Wdata  = []
    Sdata  = []
    Edata  = []
    for d in range(data.D):
        eps, Ws, sigmas = get_W_sigma_of_epsilon(
            data.Xs[d],
            data.Ys[d],
            data.Es[d],
            gridexp   = gridexp,
            show_plot = show_plot,
            epsilons  = epsilons)
        Edata.append(eps)
        Wdata.append(Ws)
        Sdata.append(sigmas)
        # try fitting
        pf_W2    = polyfit(eps, Ws**2, 1, full = True)
        pf_sigma = polyfit(eps, sigmas, 2, full = True)
        R2_W2    = calculate_R2(Ws**2, pf_W2[1])
        R2_sigma = calculate_R2(sigmas, pf_sigma[1])
        if R2_W2 > R2thrs:
            print('Accepted direction #{} W-fit with R2={}'.format(d, R2_W2))
            Wfunc = pf_W2[0]
        else:
            print('Rejected direction #{} W-fit with R2={}'.format(d, R2_sigma))
            Wfunc = None
        #end if
        if R2_sigma > R2thrs:
            print('Accepted direction #{} sigma-fit with R2={}'.format(d, R2_W2))
            Sfunc = pf_sigma[0]
        else:
            print('Rejected direction #{} sigma-fit with R2={}'.format(d, R2_sigma))
            Sfunc = None
        #end if
        Wfuncs.append(Wfunc)
        Sfuncs.append(Sfunc)
        if show_plot:
            f, ax = plt.subplots()
            ax.plot(eps, Ws, 'rx')
            ax.plot(eps, abs(polyval(pf_W2[0], eps))**0.5, 'r-')
            ax.set_ylabel('W_opt')
            ax.set_xlabel('epsilon')
            ax.set_title('linesearch #{}'.format(d))
            f, ax = plt.subplots()
            ax.plot(eps, sigmas, 'bx')
            ax.plot(eps, polyval(pf_sigma[0], eps), 'b-')
            ax.set_ylabel('sigma_opt')
            ax.set_xlabel('epsilon')
            ax.set_title('linesearch #{}'.format(d))
        #end if
    #end for
    data.W_of_epsilon     = Wfuncs
    data.sigma_of_epsilon = Sfuncs
    if get_data:
        return Edata, Wdata, Sdata
    #end if
#end def


# WIP
def bias_of_R(
    data,
    Ds   = None,
    Rmin = 1e-5,
):
    pfn      = data.pfn
    pts      = data.pts
    if Ds is None:
        Ds = range(data.D)
    #end if

    biases = []
    for d in Ds:
        x_n      = data.shifts[d]
        y_n      = data.PES[d]
        Rmax     = min(max(-x_n[x_n < 0]), max(x_n))
        R        = Rmin
        Bs       = []
        Rs       = []
        first    = True
        while R < Rmax:
            x_r      = linspace(-R, R, pts)
            x_r, y_r = interpolate_grid(x_n, y_n, x_r)
            y, x, p  = get_min_params(x_r, y_r, pfn)
            if first:
                B0 = x  # systematic bias
                first = False
                B = 0
            else:
                B = x - B0
            #end if
            Bs.append(B)
            Rs.append(R)
            R += min(Rmax / 100, 0.1)
        #end while
        biases.append((Bs, Rs))
    #end for
    data.BR = biases
    return biases
#end def


# WIP
def R_sigma_mesh(
    data,
    Ds         = None,
    Rmin       = 1e-5,
    generate   = 1000,
    num_sigma  = 11,
    num_R      = 11,
):

    pfn      = data.pfn
    pts      = data.pts
    fraction = data.fraction
    if Ds is None:
        Ds = range(data.D)
    #end if

    Xs = []
    Ys = []
    Es = []
    for d in Ds:
        try:
            B, R = data.BR[d]
        except:
            B, R = bias_of_R(data, Ds=[d])[0]
        #end try
        B_in      = interp1d(R, B)
        G         = random.randn(generate, pts)
        Lambda    = data.Lambda[d]
        x_n       = data.shifts[d]
        y_n       = data.PES[d]
        Rmax      = max(R)
        R         = Rmax
        Rs        = []
        sigma_min = 0
        sigma_max = 0.5 * Lambda * Rmax**2
        sigmas    = linspace(sigma_min, sigma_max, num_sigma)
        Err_mesh  = []
        while R > Rmin:
            bias = B_in(0.99999 * R)
            Err_sigma = []
            for sigma in sigmas:
                x_r      = linspace(-R, R, pts)
                x_r, y_r = interpolate_grid(x_n, y_n, x_r)
                xdata    = []
                for g in G:
                    y, x, p = get_min_params(x_r, y_r + sigma * g, pfn)
                    xdata.append(x)
                #end for
                Aave, Aerr = get_fraction_error(array(xdata) - bias, fraction = fraction)
                Err = Aerr + abs(bias)
                Err_sigma.append(Err)
            #end for
            Rs.append(R)
            R -= Rmax / num_R
            Err_mesh.append(Err_sigma)
        #end while
        X, Y = meshgrid(Rs, sigmas)
        Xs.append(X)
        Ys.append(Y)
        Es.append(Err_mesh)
    #end for
    return Xs, Ys, Es
#end def


def get_W_sigma_of_epsilon(
    X,                # W     mesh
    Y,                # sigma mesh
    E,                # error mesh
    gridexp   = 4.0,  # polynomial grid spacing for better resolution at small error values
    show_plot = False,
    epsilons  = None,
    zoom_in   = 1.0,
):

    if epsilons is None:
        epsilons = linspace(
            (amin(E[~isnan(E)]) + 1e-7)**(1.0 / gridexp),
            0.99 * amax(E[~isnan(E)])**(1.0 / gridexp),
            201)**gridexp
    else:
        epsilons = epsilons.copy()
    #end if
    Xi, Yi, Ei = interpolate_error_grid(X, Y, E, zoom_in)

    f, ax     = plt.subplots()
    Ws       = []
    sigmas   = []
    for epsilon in epsilons:
        W_opt     = 0.0
        sigma_opt = 0.0
        try:
            ct1 = ax.contour(Xi, Yi, Ei, [epsilon])
        except:
            epsilons = epsilons[0:len(Ws)]
            break
        #end try
        for j in range(len(ct1.allsegs)):
            for ii, seg in enumerate(ct1.allsegs[j]):
                if not len(seg) == 0:
                    i_opt = argmax(seg[:, 1])
                    if seg[i_opt, 1] > sigma_opt:
                        W_opt     = seg[i_opt, 0]
                        sigma_opt = seg[i_opt, 1]
                    #end if
                #end if
            #end for
        #end for
        if W_opt / max(X[0, :]) == 1.0 or isnan(W_opt):
            epsilons = epsilons[0:len(Ws)]
            break
        #end if
        if sigma_opt / max(Y[:, 0]) == 1.0 or isnan(sigma_opt):
            epsilons = epsilons[0:len(Ws)]
            break
        #end if
        Ws.append(W_opt)
        sigmas.append(sigma_opt)
    #end for
    if not show_plot:
        plt.close(f)
    #end if
    return epsilons, array(Ws), array(sigmas)
#end def


# Function to interpolate the W-sigma error data
#   Can be sensitive near saturation boundaries, and requires more testing to be reliable
def interpolate_error_grid(X, Y, E, zoom_in = 1):
    if zoom_in == 1:
        return X, Y, E
    #end if
    x = X[0]
    Xi = zoom(X, zoom_in)
    Yi = zoom(Y, zoom_in)
    yi = Yi[:, 0]

    # monotonicity more important along y, so interpolate that way with pchip first
    Ey = []
    for i, x_this in enumerate(x):
        y_this  = Y[:, i]
        E_this  = E[:, i]
        yi_this = Yi[:, i]
        Ey_this = pchip_interpolate(y_this, E_this, yi_this)
        Ey.append(Ey_this)
    #end for
    Ey = array(Ey).T
    # next, pchip interpolate along x
    Ei = []
    for i, y_this in enumerate(yi):
        x_this  = X[0]  # should all be the same
        xi_this = Xi[i]
        Ey_this = Ey[i]
        Ei_this = pchip_interpolate(x_this, Ey_this, xi_this)
        Ei.append(Ei_this)
    #end for
    Ei = array(Ei)
    return Xi, Yi, Ei
#end def


# Heuristic method to find the cost-optimized epsilond (an array of direction tolerances)
# Overview:
#   scan variations of epsilond
#     scale up epsilond until any parameter error hits the tolerance, estimate cost
#   choose the (re-scaled) epsilond with the lowest cost (~the one which can be scaled up the most)
def optimize_epsilond_heuristic_cost(
    data,
    epsilon,
    fraction,
    generate = 1000,
    A_min    = -0.2,  # compatibility; more natural choice is -1.0
    A_max    = 0.2,  # compatibility; more natural choice is 1.0
    A_num    = 11,  # compatibility; higher number probably beneficial, if scanning [-1,1]
):
    if fraction is None:
        fraction = data.fraction
    #end if
    epsilon = array(epsilon)

    def get_epsilond(A, sigma):
        if isscalar(sigma):
            epsilonp = array(data.D * [sigma])
        else:
            epsilonp = sigma
        #end if
        #return abs( (A*data.U + (1-A)*data.U**2) @ epsilonp)
        return abs((A * data.U + (1 - abs(A)) * data.U**2) @ epsilonp)
        # this would be formally correct, but leads in practice to pathologies (inevitably small epsilond in some directions)
        #return abs( (A*data.U + (1-A)*linalg.inv(data.U.T**2)) @ epsilonp)
    #end def

    As = linspace(A_min, A_max, A_num)
    cost_opt = 1.0e99
    cost = 0.0

    # optimize epsilond a fraction that evens out parameter errors
    delta    = 0.1
    epsilond = None
    for A in As:
        coeff    = 0.0
        for n in range(100):  # increase noise in finite steps (needs better heuristics)
            coeff          += delta
            epsilond_this   = get_epsilond(A, coeff * epsilon)
            diff, cost_this = validate_error_targets(
                data,
                epsilon,
                fraction,
                generate,
                epsilond = epsilond_this,
                get_cost = True)
            if not all(array(diff) < 0.0):
                break
            #end if
            cost      = cost_this
            epsilond  = epsilond_this.copy()
        #end for
        if cost < cost_opt:
            A_opt        = A
            cost_opt     = cost
            epsilond_opt = epsilond
        #end if
    #end for
    print('Optimized epsilond, A_opt: {}, cost={}:'.format(A_opt, cost_opt))
    print(epsilond_opt)

    return epsilond_opt
#end def


# Function to return epsilon based on thermal equipartition
def get_epsilon_thermal(data, temperature):
    Lambda = data.hessian.diagonal()
    epsilon = []
    for k in Lambda:
        epsilon.append((temperature / k)**0.5)
    #end for
    epsilon = array(epsilon)
    return epsilon
#end def


# Function to return epsilond based on thermal equipartition
def get_epsilond_thermal(data, temperature):
    Lambda = data.Lambda
    epsilond = []
    for k in Lambda:
        epsilond.append((temperature / k)**0.5)
    #end for
    epsilond = array(epsilond)
    data.temperature = temperature
    return epsilond
#end def


# Function to optimize thermal epsilond to meet tolerances epsilon by gradually increasing temperature
#   Considers all parameters, but a way to enforce only a subset of parameter tolerances is to set the rest of them high
def optimize_epsilond_thermal(data, epsilon, fraction, generate, T_step = 0.00001, T_max = 0.1):
    T = 0
    epsilond_opt = None
    while T < T_max:
        epsilond = get_epsilond_thermal(data, T)
        T       += T_step
        diff     = validate_error_targets(data, epsilon, fraction, generate, epsilond = epsilond)
        if not all(array(diff) < 0.0):
            print(diff)
            break
        #end if
        epsilond_opt = epsilond
        print('Temperature: {:<f}, max_diff: {:<f}'.format(T, max(diff)))
    #end while
    if epsilond_opt is None:
        print('Warning: thermal optimization failed! Lower starting temperature from {}. Latest diff:'.format(T_step))
        print(diff)
    #end if
    return epsilond_opt
#end def


# A wrapper function to optimize the line-search parameters (W,sigma) of a given instance of IterationData
#   Important parameters: epsilon (parameter tolerances) or temperature, optimizer
# Computes and stores predicted parameter errors in data.params_next_err
def optimize_window_sigma(
    data,
    epsilon     = 0.01,
    temperature = 0,    # alternative to epsilon
    epsilond    = None,
    show_plot   = False,
    fraction    = None,
    optimizer   = optimize_epsilond_heuristic_cost,  # can also be e.g. optimize_epsilond_broyden1
    generate    = 0,  # default: use existing data
    verbose     = False,
):

    if fraction is None:
        fraction = data.fraction
    #end if

    if epsilond is None:
        if temperature > 0:
            epsilond = get_epsilond_thermal(data, temperature)
        else:
            epsilond = optimizer(data, epsilon, fraction, generate)
        #end if
    #end if

    # finally, set optimal windows and sigmas
    windows    = []
    noises     = []
    bias_corrs = None  # not for the moment
    for d in range(data.D):
        try:
            W     = abs(polyval(data.W_of_epsilon[d], epsilond[d]))**0.5
            sigma = abs(polyval(data.sigma_of_epsilon[d], epsilond[d]))
        except:
            X = data.Xs[d]
            Y = data.Ys[d]
            E = data.Es[d]
            W, sigma, err = optimize_linesearch(X, Y, E, epsilon = epsilond[d], title = '#%d' % d, show_plot = show_plot)
        #end try
        windows.append(W)
        noises.append(sigma)
    #end for
    data.params_next_err = validate_error_targets(data, None, fraction, generate, epsilond, fractional=False)
    data.bias_corrs = bias_corrs
    data.epsilond   = epsilond
    data.noises     = noises
    data.windows    = windows
    data.fraction   = fraction
    data.is_noisy   = True
#end def


# If regression models for W and sigma are unavailable (load_of_epsilon not successfully done), fall back to using the contour approach
def optimize_linesearch(
    X,
    Y,
    E,
    epsilon     = 0.01,
    show_plot   = True,
    show_levels = 15,
    savefig     = None,
    title       = '',
    output      = False,
):
    f, ax  = plt.subplots()
    errors = False
    levels = linspace(0, 2 * epsilon, 15)
    ctf   = ax.contourf(X, Y, E, levels)
    ct1   = ax.contour(X, Y, E, [epsilon], colors=['k'])

    # find the optimal points
    W_opt     = 0.0
    sigma_opt = 0.0
    for j in range(len(ct1.allsegs)):
        for ii, seg in enumerate(ct1.allsegs[j]):
            if not len(seg) == 0:
                i_opt = argmax(seg[:, 1])
                if seg[i_opt, 1] > sigma_opt:
                    W_opt     = seg[i_opt, 0]
                    sigma_opt = seg[i_opt, 1]
                #end if
            #end if
        #end for
    #end for
    if sigma_opt == 0 or W_opt == 0:
        print('Warning: optimal points not found! Lower W and sigma ranges!')
        errors = True
    else:
        ax.plot(W_opt, sigma_opt, 'kx', label = 'W=%f, sigma=%f' % (W_opt, sigma_opt))
    #end if
    ax.set_xlabel('Energy window')
    ax.set_ylabel('Input noise')
    ax.legend(fontsize = 8)
    ax.set_title(title + ' total error, epsilond=%f' % epsilon)
    plt.subplots_adjust(left = 0.2, right = 0.98)
    f.colorbar(ctf)

    if W_opt / max(X[0, :]) == 1.0:
        print('Warning: bad resolution of W optimization. Increase W range from %f!' % amax(X))
        errors = True
    #end if
    if sigma_opt / max(Y[:, 0]) == 1.0:
        print('Warning: bad resolution of sigma optimization. Increase sigma range from %f!' % amax(Y))
        errors = True
    #end if

    if savefig is not None:
        plt.savefig(savefig)
    #end if

    if output:
        print('optimal W:     ' + str(W_opt))
        print('optimal sigma: ' + str(sigma_opt))
        print('relative cost: ' + str(sigma_opt**-2))
    #end if
    if not show_plot:
        plt.close(f)
    #end if

    return W_opt, sigma_opt, errors
#end def


# Function to optimize epsilond using Brouden's method (currently instable and untested)
def optimize_epsilond_broyden1(data, epsilon, fraction, generate, verbose = False):
    if fraction is None:
        fraction = data.fraction
    #end if
    epsilond0 = data.D * [epsilon]
    validate_epsilond = partial(validate_error_targets, data, epsilon, fraction, generate)
    epsilond_opt = broyden1(validate_epsilond, epsilond0, f_tol = 1e-3, verbose = verbose)
    return epsilond_opt
#end def


# Obsolete: Function to heuristically optimize epsilond based on deviation from target values
def optimize_epsilond_heuristic(data, epsilon, fraction, generate):
    if fraction is None:
        fraction = data.fraction
    #end if

    def get_epsilond(A, sigma):
        if isscalar(sigma):
            epsilonp = array(data.D * [sigma])
        else:
            epsilonp = sigma
        #end if
        return abs((A * data.U + (1 - A) * data.U**2) @ epsilonp)
        #return abs( (A*data.U + (1-A)*linalg.inv(data.U.T**2)) @ epsilonp)
    #end def

    #As = linspace(0.0,1.0,11)
    As = linspace(-0.2, 0.2, 11)

    # optimize epsilond ab fraction that evens out parameter errors
    varAs = []
    for A in As:
        epsilond = get_epsilond(A, epsilon)
        diff     = validate_error_targets(data, epsilon, fraction, generate, epsilond)
        varAs.append(var(diff))
    #end for
    A_opt = As[argmin(array(varAs))]

    # optimize input noise prefactor
    delta = 0.1
    coeff = 0.0
    for n in range(100):
        coeff          += delta
        epsilond_this   = get_epsilond(A_opt, coeff * epsilon)
        diff, cost_this = validate_error_targets(data, epsilon, fraction, generate, epsilond = epsilond_this, get_cost = True)
        if not all(array(diff) < 0.0):
            break
        #end if
        cost_opt      = cost_this
        epsilond_opt  = epsilond_this.copy()
    #end for
    if n == 1:
        print('Warning: epsilond broke at first try!')
    else:
        print('Cost-optimized epsilond, A_opt:{}, cost={}:'.format(A_opt, cost_opt))
        print(epsilond_opt)
    #end if
    return epsilond_opt
#end def


# Function to simulate mixing of parameter errors from the line-search ensemble epsilond.
# Return the observed parameter errors in absolute (actual errors) or fractional units (easy to monitor; negative value means below tolerance)
def validate_error_targets(
    data,        # Iteration data
    epsilon,     # target parameter accuracy
    fraction,    # statistical fraction
    generate,    # use old random data or create new
    epsilond    = None,   # tolerances per search direction
    windows     = None,   # set of noises
    noises      = None,   # set of windows
    get_cost    = False,  # estimate cost
    fractional  = True,   # return error in fractional form
):
    use_epsilond = epsilond is not None
    use_W_sigma  = windows is not None and noises is not None

    Ds     = []
    cost   = 0.0
    for d in range(data.D):
        if use_W_sigma:
            W_opt     = windows[d]
            sigma_opt = noises[d]
        elif use_epsilond:
            eps = abs(epsilond[d])
            try:
                W_opt     = abs(polyval(data.W_of_epsilon[d], eps))**0.5
                sigma_opt = abs(polyval(data.sigma_of_epsilon[d], eps))
            except:
                X = data.Xs[d]
                Y = data.Ys[d]
                E = data.Es[d]
                W_opt, sigma_opt, err = optimize_linesearch(X, Y, E, epsilon = epsilond[d], show_plot = False)
            #end try
        else:  # use raw data
            W_opt     = data.windows[d]
            sigma_opt = data.noises[d]
        #end if
        if generate > 0:
            Gs = generate
        else:
            Gs = data.Gs[d]
        #end if
        B0 = data.Bs[d][0]
        D = get_search_distribution(
            x_n       = data.Dshifts[d],
            y_n       = data.PES[d],
            H         = data.Lambda[d],
            W_opt     = W_opt,
            sigma_opt = sigma_opt,
            pfn       = data.pfn,
            pts       = data.pts,
            Gs        = Gs,
            x_0       = B0,  # compensate for relaxation bias
        )
        Ds.append(D)
        cost += data.pts * sigma_opt**-2
    #end for
    Ds = array(Ds).T
    # propagate search error
    errs = []
    for p in range(Ds.shape[1]):
        ave, err = get_fraction_error((Ds @ data.U)[:, p], fraction = fraction)
        errs.append(abs(ave) + err)  # conservative summation of errors
    #end for

    # return fractional error
    if fractional:
        diff = array(errs) / epsilon - 1.0
        if get_cost:
            return diff, cost
        else:
            return diff
        #end if
    else:
        return array(errs)
    #end if
#end def


# Returns a resampled distribution of line-search results for error analysis.
# The line-search is set up based on interpolated quantities
def get_search_distribution(
    x_n,
    y_n,
    H,
    W_opt,
    sigma_opt,
    pfn,
    pts,
    Gs  = 1000,
    x_0 = 0.0,
):
    xy_in = interp1d(x_n, y_n, kind = 'cubic')
    if isscalar(Gs):
        Gs = random.randn(Gs, pts)
    #end if
    generate = Gs.shape[0]

    R      = W_to_R(W_opt, H)
    R      = max(min(x_n), R)
    R      = min(max(x_n), R)
    x_r    = linspace(-R, R, pts)
    y_r    = xy_in(x_r)

    xdata = []
    for n in range(generate):
        y_min, x_min, pf = get_min_params(x_r, y_r + sigma_opt * Gs[n], pfn)
        xdata.append(x_min - x_0)
    #end for
    dxdata = array(xdata)
    return dxdata
#end def


# Function to analyze and print out some basic information about the error scan results
def error_scan_diagnostics(data, steps_times_error2 = None):
    # cost
    cost_max   = min(array(data.noises))**-2
    cost_tot_d = sum(array(data.noises)**-2)
    cost_tot   = (data.pts - 1) * cost_tot_d + cost_max

    print('Error scan diagnostics')
    print('  polyfit degree: {}'.format(data.pfn))
    print('  pts:            {}'.format(data.pts))
    try:
        print('  temperature:    {} Ry'.format(data.temperature))
    except:
        pass
    #end try
    print('')

    # line-search parameters
    if steps_times_error2 is None:
        print('{:10s} {:15s} {:15s} {:15s} {:15s} {:15s}'.format('direction', 'target', 'window (Ry)', 'R (p-unit)', 'noise (Ry)', 'rel. cost (%)'))
        for d in range(data.D):
            W         = data.windows[d]
            H         = data.Lambda[d]
            R         = W_to_R(W, H)
            sigma     = data.noises[d]
            cost      = sigma**-2 / cost_tot_d * 100
            epsilond  = data.epsilond[d]
            print('{:<10d} {:<15f} {:<15f} {:<15f} {:<15f} {:<04.2f}'.format(d, epsilond, W, R, sigma, cost))
        #end for
    else:
        print('{:10s} {:15s} {:15s} {:15s} {:15s} {:15s} {:15s}'.format('direction', 'target', 'window (Ry)', 'R (p-unit)', 'noise (Ry)', 'rel. cost (%)', 'QMC steps'))
        max_steps = 0
        tot_steps = 0
        for d in range(data.D):
            W          = data.windows[d]
            H          = data.Lambda[d]
            R          = W_to_R(W, H)
            sigma      = data.noises[d]
            cost       = sigma**-2 / cost_tot_d * 100
            steps      = int(4 * steps_times_error2 * sigma**-2) + 1  # 4x factor from unit conversion
            tot_steps += steps * (data.pts - 1)
            max_steps  = max(max_steps, steps)
            epsilond   = data.epsilond[d]
            print('{:<10d} {:<15f} {:<15f} {:<15f} {:<15f} {:<15.2f} {:<15d}'.format(d, epsilond, W, R, sigma, cost, steps))
        #end for
        tot_steps += max_steps
    #end if

    # estimated parameter errors
    try:
        param_errs = data.params_next_err
        print('\nParameter  error')
        for p in range(data.P):
            print('{:<10d} {:<15f}'.format(p, param_errs[p]))
        #end for
    except:
        print('\nParameter error estimates not available.')
    #end try

    try:
        epsilonp = data.epsilonp
        T        = data.T
        params   = data.params
        print('\nEstimated parameters and errors errors: (T={} Ry)'.format(T))
        for p, epsilon in enumerate(epsilonp):
            print('  p{}: {:<8f} +/- {:<8f}'.format(p, params[p], epsilon))
        #end for
    except:
        pass
    #end try

    print('\ntotal relative cost: {:e}'.format(cost_tot))
    if steps_times_error2 is not None:
        print('Equivalent QMC errorbar: {:<5e} Ha'.format((steps_times_error2 / tot_steps)**0.5))
    #end if
#end def


# Evaluate the array of noises for a given epsilond
def get_noises(data, epsilond):
    noises = []
    for d in range(data.D):
        try:
            sigma = abs(polyval(data.sigma_of_epsilon[d], epsilond[d]))
        except:
            X = data.Xs[d]
            Y = data.Ys[d]
            E = data.Es[d]
            W, sigma, err = optimize_linesearch(X, Y, E, epsilon = epsilond[d], title='#%d' % d)
        #end try
        noises.append(sigma)
    return noises
#end if


# Returns the relative balance of costs across search directions, normalized to 1
def cost_balance(data, noises = None, epsilond = None, get_norm = False):
    if noises is None:
        if epsilond is None:
            noises = data.noises
        else:
            noises = get_noises(data, epsilond)
        #end if
    #end if
    noises       = array(noises)
    costs        = noises**-2
    norm         = sum(costs)
    cost_balance = costs / norm
    if get_norm:
        return cost_balance, norm
    else:
        return cost_balance
    #end if
#end def


# Function to return parameter biases (based on surrogate PES) with the given set of windows
def surrogate_parameter_biases(data):
    D_biases = []
    for d in range(data.D):
        Ws     = data.Xs[d][0]
        Bs     = data.Bs[d]
        B_in   = interp1d(Ws, Bs)
        window = data.windows[d]
        try:
            bias = B_in(window)
        except:
            print('Warning: max window exceeded in d{}: {} > {}'.format(d, window, Ws[-1]))
            bias = Bs[-1]
        #end try
        D_biases.append(bias)
    #end for
    P_biases = D_biases @ data.U
    return array(P_biases)
#end def


# Centralized function for PES ion
def interpolate_grid(x_in, y_in, x_out, kind = 'pchip'):
    if kind == 'pchip':
        # move this elsewhere
        s = argsort(array(x_in))
        y_out = pchip_interpolate(array(x_in)[s], array(y_in)[s], x_out)
    else:
        xy_in = interp1d(x_in, y_in, kind = kind)
        try:
            y_out = xy_in(x_out)
        except:
            print('Warning: interpolation failed, returning original grid')
            x_max = min([-min(x_in), max(x_in)])
            x_out = linspace(-x_max, x_max, len(x_out))
            y_out = xy_in(x_out)
        #end try
    #end if
    return x_out, y_out
#end def
