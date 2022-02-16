#!/usr/bin/env python3

from numpy import array, zeros, ones, dot, diag, transpose, sqrt, repeat
from numpy import linalg, meshgrid, polyfit, polyval, argmin, linspace, ceil
from numpy import pi, isnan, nan, mean, isscalar, roots, polyder, savetxt
from numpy import flipud, median, arccos
from nexus import PwscfAnalyzer


# Utility for printing value with uncertainty in parentheses
def print_with_error(value, error, limit = 15):
    if error == 0.0 or isnan(error):
        return str(value)
    #end if
    ex = -9
    while ceil(error * 10**(ex + 1)) > 0 and ceil(error * 10**(ex + 1)) < limit:
        ex += 1
    #end while
    errstr = str(int(ceil(error * 10**ex)))

    if ex == 1 and ceil(error) >= 1.0:
        fmt = '%' + str(ex + 2) + '.' + str(ex) + 'f'
        valstr = fmt % value
        errstr = '%1.1f' % error
    elif ex > 0:  # error is in the decimals
        fmt = '%' + str(ex + 2) + '.' + str(ex) + 'f'
        valstr = fmt % value
    else:  # error is beyond decimals
        fmt = '%1.0f'
        errstr += (-ex) * '0'
        val = round(value * 10**ex) * 10**(-ex)
        valstr = fmt % val
    #end if
    return valstr + '(' + errstr + ')'
#end def


# Wrapper for loading force-constant matrices from QE or VASP
def load_gamma_k(fname, num_prt, symmetrize = True, **kwargs):
    if fname.endswith('.fc'):  # QE
        K = load_force_constants_qe(fname, num_prt, **kwargs)
    elif fname.endswith('.hdf5'):  # VASP
        K = load_force_constants_vasp(fname, num_prt, **kwargs)
    else:
        print('Force-constant file not recognized (.fc and .hdf5 supported)')
        K = None
    #end if
    if symmetrize and K is not None:
        for k0 in range(K.shape[0]):
            for k1 in range(k0 + 1, K.shape[1]):
                val = (K[k0, k1] + K[k1, k0]) / 2
                K[k0, k1] = val
                K[k1, k0] = val
            #end for
        #end for
    #end if
    return K
#end def


# Load FC matrix in QE format
def load_force_constants_qe(fname, num_prt, dim = 3):
    K = zeros((dim * num_prt, dim * num_prt))
    with open(fname) as f:
        line = f.readline()
        while line:
            line_spl = line.split()
            if len(line_spl) == 4 and len(line_spl[3]) < 3:  # stupid way to check for integer?
                dim1 = int(line_spl[0])
                dim2 = int(line_spl[1])
                prt1 = int(line_spl[2])
                prt2 = int(line_spl[3])
                line = f.readline().split()  # gamma point is the first line
                i = (prt1 - 1) * dim + dim1 - 1
                j = (prt2 - 1) * dim + dim2 - 1
                K[i, j] = float(line[3])
            #end if
            line = f.readline()
        #end if
    #end with
    return K
#end def


# load FC matrix in VASP format
#   will need better unit conversion
def load_force_constants_vasp(fname, num_prt, dim = 3):
    import h5py
    f = h5py.File(fname, mode = 'r')

    K_raw = array(f['force_constants'])
    p2s   = array(f['p2s_map'])

    # this could probably be done more efficiently with array operations
    K = zeros((dim * num_prt, dim * num_prt))
    for prt1 in range(num_prt):
        for prt2 in range(num_prt):
            sprt2 = p2s[prt2]
            for dim1 in range(dim):
                for dim2 in range(dim):
                    i = prt1 * dim + dim1
                    j = prt2 * dim + dim2
                    K[i, j] = K_raw[prt1, sprt2, dim1, dim2]
                #end for
            #end for
        #end for
    #end for
    f.close()

    # assume conversion from eV/Angstrom**2 to Ry/Bohr**2
    eV_A2 = 27.211399 / 2 * 0.529189379**-2
    return K / eV_A2
#end def


# Load phonon modes from QE output
def load_phonon_modes(fname, num_prt, drop_modes = 0):
    ws    = []
    vects = []
    with open(fname) as f:
        line = f.readline()
        while line:
            line_spl = line.split()
            if len(line_spl) == 9 and line_spl[0] == 'freq':
                freq = float(line_spl[7])
                vect = []
                for prt in range(num_prt):
                    line = f.readline().split()  # gamma point is the first line
                    vect.append(float(line[1]))
                    vect.append(float(line[3]))
                    vect.append(float(line[5]))
                #end for
                vects.append(vect)
                ws.append(freq)
            #end if
            line = f.readline()
        #end if
    #end with
    # drop rotational and translational modes
    w = array(ws[drop_modes:])
    v = array(vects[drop_modes:][:])
    if len(v.shape) == 1:  # if only one mode
        v = array([v])
    #end if
    return array(w), array(v)
#end def


# read geometry in QE format to row format
def read_geometry(geometry_string):
    lines = geometry_string.split('\n')
    R = []
    names = []
    for line in lines:
        fields = line.split()
        if len(fields) > 3:
            names.append(fields[0])
            R.append(float(fields[1]))
            R.append(float(fields[2]))
            R.append(float(fields[3]))
        #end if
    #end for
    return array(R), names
#end def


# Print geometry in QE format
def print_qe_geometry(atoms, positions, dim = 3):
    for a, atom in enumerate(atoms):
        coords = ''
        for i in range(dim * a, dim * a + dim):
            coords += str(positions[i]).ljust(20)
        print(atom.ljust(6) + coords)
    #end for
#end def


# Print the diagonal of a force-constant matrix
def print_fc_matrix(fc, num_prt, diagonals_only = True, title = None):
    if title is not None:
        print('Force-constant matrix: ' + title)
    #end if
    dim = 3
    for i1, prt1 in enumerate(range(num_prt)):
        for i2, prt2 in enumerate(range(num_prt)):
            for r, xyz in enumerate(['x', 'y', 'z']):
                repr_str = str(xyz) + str(i1) + ' ' + str(xyz) + str(i2)
                ph_str   = str(fc[i1 * dim + r, i2 * dim + r])
                print(repr_str.ljust(16) + ph_str.ljust(16))
            #end for
        #end for
    #end for
    print('')
#end def


# w must be 1-d array of frequencies in cm-1
# vects must be 2-d array: num_freq x num_prt*dim
# masses is 1-d vector of num_prt
# there may be a better way to do this, but does it matter
def K_from_W(w, v, masses, dim = 3):
    w2 = (array(w) * 2 * 0.000004556335281)**2  # cm-1 to rydberg
    sqrt_m = sqrt(diag(repeat(masses, dim)))    # multiply by sqrt of masses
    # correctly normalize disp vectors
    v2 = v.copy() @ sqrt_m
    for row in range(v2.shape[0]):
        v2[row, :] *= 1 / linalg.norm(v2[row, :])
    #end for
    K = sqrt_m @ transpose(v2) @ diag(w2) @ v2 @ sqrt_m
    return K
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


# Important function to resolve the local minimum of a curve
#   If endpts are given, search minima from them, too
def get_min_params(x_n, y_n, pfn = 2, endpts=[]):
    pf     = polyfit(x_n, y_n, pfn)
    r      = roots(polyder(pf))
    Pmins  = list(r[r.imag == 0].real)
    if len(endpts) > 0:
        for Pmin in Pmins:
            if Pmin < min(endpts) or Pmin > max(endpts):
                Pmins.remove(Pmin)
            #end if
        #end for
        for pt in endpts:
            Pmins.append(pt)
        #end for
    #end if
    Emins = polyval(pf, array(Pmins))
    try:
        imin = argmin(Emins)
        Emin = Emins[imin]
        Pmin = Pmins[imin]
    except ValueError:
        Pmin = nan
        Emin = nan
    return Emin, Pmin, pf
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


# Estimate conservative (maximum) uncertainty from a distribution based on a percentile fraction
def get_fraction_error(data, fraction, both = False):
    if fraction < 0.0 or fraction > 0.5:
        raise ValueError('Invalid fraction')
    #end if
    data   = array(data)
    data   = data[~isnan(data)]        # remove nan
    ave    = median(data)
    data   = data[data.argsort()] - ave  # sort and center
    pleft  = abs(data[int((len(data) - 1) * fraction)])
    pright = abs(data[int((len(data) - 1) * (1 - fraction))])
    if both:
        err    = [pleft, pright]
    else:
        err    = max(pleft, pright)
    #end if
    return ave, err
#end def


# Parameter mapping utility for merging pos and cell arrays
def merge_pos_cell(pos, cell):
    posc = array(list(pos.flatten()) + list(cell.flatten()))  # generalized position vector: pos + cell
    return posc
#end def


# assume 3x3
def detach_pos_cell(posc, num_prt = None, dim = 3, reshape = True):
    posc = posc.reshape(-1, dim)
    if num_prt is None:
        pos  = posc[:-dim].flatten()
        cell = posc[-dim:].flatten()
    else:
        pos  = posc[:num_prt].flatten()
        cell = posc[num_prt:].flatten()
    #end if
    if reshape:
        return pos.reshape(-1, 3), cell.reshape(-1, 3)
    else:
        return pos, cell
    #end if
#end def


# Read relaxed structure from SCF with Nexus PwscfAnalyzer
def get_relax_structure(
    path,
    suffix     = 'relax.in',
    pos_units  = 'B',
    relax_cell = False,
    dim        = 3,
    celldm1    = 1.0,  # multiply cell by constant (in B)
):
    relax_path = '{}/{}'.format(path, suffix)
    try:
        relax_analyzer = PwscfAnalyzer(relax_path)
        relax_analyzer.analyze()
    except:
        print('No relax geometry available: run relaxation first!')
    #end try

    # get the last structure
    eq_structure = relax_analyzer.structures[len(relax_analyzer.structures) - 1]
    pos_relax    = eq_structure.positions.flatten()
    if relax_cell:
        cell_relax = eq_structure.axes.flatten() * celldm1
        pos_relax = array(list(pos_relax) + list(cell_relax))  # generalized position vector: pos + cell
    #end if
    if pos_units == 'A':
        pos_relax *= 0.5291772105638411
    #end if
    return pos_relax
#end def


# Compute parameter Hessian by using either JAX or finite difference
def compute_hessian(jax_hessian = False, eps = 0.001, **kwargs):
    if jax_hessian:
        print('Computing parameter Hessian with JAX')
        hessian_delta = compute_hessian_jax(**kwargs)
    else:
        print('Computing parameter Hessian with finite difference')
        hessian_delta = compute_hessian_fdiff(eps = eps, **kwargs)
    #end if
    return hessian_delta
#end def


# Compute parameter Hessian using JAX
def compute_hessian_jax(
    hessian_pos,
    params_to_pos,
    pos_to_params,
    pos,
    **kwargs
):
    from jax import grad
    p0 = pos_to_params(pos)
    gradfs = []
    for i, r in enumerate(pos):
        def pp(p):
            return params_to_pos(p)[i]
        #end def
        gradf = grad(pp)
        gradfs.append(gradf(p0))
    #end for
    gradfs = array(gradfs)
    hessian_delta = gradfs.T @ hessian_pos @ gradfs
    savetxt('H_delta.dat', hessian_delta)
    return hessian_delta
#end def


# Compute parameter Jacobian using finite difference
def compute_jacobian_fdiff(params_to_pos, params, eps = 0.001):
    jacobian = []
    pos_orig = params_to_pos(params)
    for p in range(len(params)):
        p_this     = params.copy()
        p_this[p] += eps
        pos_this   = params_to_pos(p_this)
        jacobian.append((pos_this - pos_orig) / eps)
    #end for
    jacobian = array(jacobian).T
    return jacobian
#end def


# Compute Parameter Hessian based on finite-differnce Jacobian
def compute_hessian_fdiff(
    hessian_pos,
    params_to_pos,
    pos_to_params,
    pos,
    eps  = 0.001,
    **kwargs,
):
    params   = pos_to_params(pos)
    jacobian = compute_jacobian_fdiff(params_to_pos, params = params, eps = eps)
    hessian_delta = jacobian.T @ hessian_pos @ jacobian
    return hessian_delta
#end try


# Centralized printing utility for the parameter Hessian
def print_hessian_delta(hessian_delta, U, Lambda, roundi = 3):
    print('Parameters Hessian (H_Delta)')
    print(hessian_delta.round(roundi))
    print('')
    print('Eigenvectors (U; params x directions):')
    print(U.round(roundi))
    print('')
    print('Eigenvalues (Lambda):')
    print(Lambda.round(roundi))
    print('')
#end def


# Centralized printing utility for the relaxed structure
def print_relax(elem, pos_relax, params_relax, dim = 3):
    print('Relaxed geometry (non-symmetrized):')
    print_qe_geometry(elem, pos_relax, dim)
    print('Parameter values (non-symmetrized):')
    for p, pval in enumerate(params_relax):
        print(' #{}: {}'.format(p, pval))
    #end for
#end def


def relax_diagnostics(p_vals, p_labels = None):
    print('Relaxed parameters:')
    if p_labels is None:
        p_labels = ['#{}'.format(i) for i in range(len(p_vals))]
    #end if
    for p_val, p_label in zip(p_vals, p_labels):
        print('  {}: {}'.format(p_label, p_val))
    #end for
#end def


# Matrix utilities for modeling polyfit errors
def calculate_X_matrix(x_n, pfn):
    X = []
    for x in x_n:
        row = []
        for pf in range(pfn + 1):
            row.append(x**pf)
        #end for
        X.append(row)
    #end for
    X = array(X)
    return X
#end def


def calculate_F_matrix(X):
    F = linalg.inv(X.T @ X) @ X.T
    return F
#end def


# Model statistical bias of 2 or 3 degree polyfits, accurate to O(sigma**4)
def model_statistical_bias(pf, x_n, sigma):
    pfn = len(pf) - 1
    # opposite index convention
    p   = flipud(pf)
    X = calculate_X_matrix(x_n, pfn)
    F = calculate_F_matrix(X)
    s2 = sigma**2 * (F @ F.T)
    if pfn == 2:
        bias = (s2[1, 2] / p[2]**2 - s2[2, 2] * p[1] / p[2]**3) / 2
    elif pfn == 3:
        z = (p[2]**2 - 3 * p[1] * p[3])**0.5
        b11 = s2[1, 1] * (-3 / 4 * p[3] * z**-3) / 2
        b22 = s2[2, 2] * (1 / 3 / p[3] / z - p[2]**2 / 3 / p[3] / z**3) / 2
        b33 = s2[3, 3] * (1 / 3 / p[3]) * (2 / p[3]**2 * (z - p[2]) + 3 * p[1] / p[3] / z - 9 / 4 * p[1]**2 / z**3) / 2
        b12 = s2[1, 2] * (p[2] / z**3) / 2
        b13 = s2[1, 3] * (-3 / 2 * p[1] / z**3) / 2
        b23 = s2[2, 3] * (p[1] * p[2] / p[3] / z**3 - 2 * (p[2] / z - 1) / 3 / p[3]**2) / 2
        bias = b11 + b22 + b33 + b12 + b13 + b23
    else:  # no correction known
        bias = 0.0
    #end if
    return bias
#end def


# Plotting utility for producing parameter coupling heatmaps
def plot_U_heatmap(ax, U, sort = True, cmap = 'RdBu', labels = True, reorder = False):
    U_cp  = U.copy()
    D     = U.shape[0]
    ticks = range(D)
    if sort:
        U_sorted    = []
        yticklabels = []
        rows        = list(range(D))
        for d in range(D):
            # find the largest abs value of d from the remaining rows
            i = abs(U_cp[rows[d:], d]).argmax()
            rows = rows[:d] + [rows.pop(d + i)] + rows[d:]
        #end for
        U_sorted = array(U_cp[rows])
        yticklabels = rows
    else:
        U_sorted    = U_cp
        yticklabels = [str(d) for d in range(D)]
    #end if
    cb = ax.imshow(U_sorted, cmap=cmap, vmin = -1.0, vmax = 1.0, aspect = 'equal')
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_yticklabels(yticklabels)
    if labels:
        ax.set_ylabel('Directions')
        ax.set_xlabel('Parameters')
    #end if
    return cb
#end def


# Check consistency of parameter mappings
def check_mapping_consistency(
    p_init,
    pos_to_params,
    params_to_pos,
    tol = 1e-10,
):
    if any(abs(pos_to_params(params_to_pos(p_init)) - p_init) > tol):
        print('Trouble with consistency!')
        return False
    else:
        return True
    #end if
#end def


# distance between two atomic coordinates
def distance(r0, r1):
    r = linalg.norm(r0 - r1)
    return r
#end def


# bond angle between r0-rc and r1-rc bonds
def bond_angle(r0, rc, r1, units = 'ang'):
    v1 = r0 - rc
    v2 = r1 - rc
    cosang = dot(v1, v2) / linalg.norm(v1) / linalg.norm(v2)
    ang = arccos(cosang) * 180 / pi if units == 'ang' else arccos(cosang)
    return ang
#end def


def mean_distances(pairs):
    rs = []
    for pair in pairs:
        rs.append(distance(pair[0], pair[1]))
    #end for
    return array(rs).mean()
#end def


# propagate errors of params to nparams based on mapping Jacobian obtained
# from finite differences
def params_to_nparams_error(
    params,
    param_errs,
    params_to_pos,
    pos_to_nparams,
    fdiff = 0.001,
):
    np_errs = []
    nparams = pos_to_nparams(params_to_pos(params))
    for p, param in enumerate(params):
        fparams     = params.copy()
        fparams[p] += fdiff
        dnparams     = (pos_to_nparams(params_to_pos(fparams)) - nparams) / fdiff
        np_errs.append(dnparams * param_errs[p])
    #end for
    np_errs2 = (array(np_errs))**2
    np_errs = np_errs2.sum(axis = 0)**0.5
    return nparams, np_errs
#end def


# generate finite-difference
def generate_fdiff_jobs(
    params,
    params_to_pos,
    pos_to_params,
    get_job,
    dp   = 0.01,
    path = 'fdiff/'
):
    # eqm
    pos_eqm = params_to_pos(params)
    eqm_job = get_job(pos_eqm, path = path + 'scf')
    jobs = eqm_job
    diffs = [len(params) * [0.]]
    if isscalar(dp):
        dp = len(params) * [dp]
    #end if

    def shift_params(p_ids, dps):
        nparams = params.copy()
        string = 'scf'
        for p, dp in zip(p_ids, dps):
            nparams[p] += dp
            string += '_p{}'.format(p)
            if dp > 0:
                string += '+'
            #end if
            string += '{}'.format(dp)
        #end def
        return get_job(params_to_pos(nparams), path = path + string), list(nparams - params)
    #end def

    for p0, param0 in enumerate(params):
        job, dparams = shift_params([p0], [+dp[p0]])
        jobs += job
        diffs += [dparams]
        job, dparams = shift_params([p0], [-dp[p0]])
        jobs += job
        diffs += [dparams]
        for p1, param1 in enumerate(params):
            if p1 <= p0:
                continue
            #end if
            job, dparams = shift_params([p0, p1], [+dp[p0], +dp[p1]])
            jobs += job
            diffs += [dparams]
            job, dparams = shift_params([p0, p1], [+dp[p0], -dp[p1]])
            jobs += job
            diffs += [dparams]
            job, dparams = shift_params([p0, p1], [-dp[p0], +dp[p1]])
            jobs += job
            diffs += [dparams]
            job, dparams = shift_params([p0, p1], [-dp[p0], -dp[p1]])
            jobs += job
            diffs += [dparams]
        #end for
    #end for
    return array(diffs), jobs
#end def


# analyze finite-difference
def analyze_fdiff_jobs(
    diffs,
    params,
    energies = None,
    jobs = None,
):
    if energies is None:
        Es = []
        for job in jobs:
            a = job.locdir + '/' + job.infile
            pa = PwscfAnalyzer(a)
            pa.analyze()
            Es.append(pa.E)
        #end for
        energies = Es
    #end if
    P = len(params)
    if P == 1:  # for 1-dimensional problems
        pf = polyfit(diffs[:, 0], energies, 2)
        hessian = array([[pf[0]]])
    else:
        hessian = zeros((P, P))
        pfs = P * [[]]
        for p0, param0 in enumerate(params):
            for p1, param1 in enumerate(params):
                if p1 <= p0:
                    continue
                #end if
                # filter out the values where other parameters were altered
                ids = ones(len(diffs), dtype=bool)
                for p in range(P):
                    if p == p0 or p == p1:
                        continue
                    #end if
                    ids = ids & (abs(diffs[:, p]) < 1e-10)
                #end for
                XY = diffs[ids]
                E = energies[ids]
                X = XY[:, 0]
                Y = XY[:, 1]
                pf = bipolyfit(X, Y, E, 2, 2)
                hessian[p0, p1] = pf[4]
                hessian[p1, p0] = pf[4]
                pfs[p0].append(2 * pf[2])
                pfs[p1].append(2 * pf[6])
            #end for
        #end for
        for p0 in range(P):
            hessian[p0, p0] = mean(pfs[p0])
        #end for
    #end if
    return hessian
#end def


# get displaced parameter sets
def fetch_parameters(data):
    eqm = data.params
    params = []
    for d in range(data.D):
        params_d = []
        for s in data.shifts[d]:
            p_this = eqm.copy() + data.U[d] * s
            params_d.append(p_this)
        #end for
        params.append(params_d)
    #end for
    return params
#end def
