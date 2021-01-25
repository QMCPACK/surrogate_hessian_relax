#!/usr/bin/env python3

from numpy import array,loadtxt,zeros,dot,diag,transpose,sqrt,repeat,linalg,reshape,meshgrid,poly1d,polyfit,polyval,argmin,linspace,random,ceil,diagonal,amax,argmax,pi,isnan,nan,mean,var,amin,isscalar,roots,polyder,savetxt,flipud,delete
from math import log10


def print_with_error( value, error, limit=15 ):

    if error==0.0 or isnan(error):
        return str(value)
    #end if

    ex = -9
    while ceil( error*10**(ex+1) ) > 0 and ceil( error*10**(ex+1)) < limit:
        ex += 1
    #end while
    errstr = str(int(ceil(error*10**ex)))

    if ex==1 and ceil(error) >= 1.0:
        fmt = '%'+str(ex+2)+'.'+str(ex)+'f'
        valstr = fmt % value
        errstr = '%1.1f' % error
    elif ex>0: # error is in the decimals
        fmt = '%'+str(ex+2)+'.'+str(ex)+'f'
        valstr = fmt % value
    else:    # error is beyond decimals
        fmt = '%1.0f'
        errstr += (-ex)*'0'
        val = round(value*10**ex)*10**(-ex)
        valstr = fmt % val
    #end if
    return valstr+'('+errstr+')'
#end def


# load force-constants
def load_gamma_k(fname, num_prt, **kwargs):
    if fname.endswith('.fc'): # QE
        K = load_force_constants_qe(fname, num_prt, **kwargs)
    elif fname.endswith('.hdf5'): # VASP
        K = load_force_constants_vasp(fname, num_prt, **kwargs)
    else:
        print('Force-constant file not recognized (.fc and .hdf5 supported)')
        K = None
    #end if
    return K
#end def


def load_force_constants_qe(fname, num_prt, dim=3):
    K = zeros((dim*num_prt,dim*num_prt))
    with open(fname) as f:
        line = f.readline()
        while line:
            line_spl = line.split()
            if len(line_spl)==4 and len(line_spl[3])<3: # stupid way to check for integer?
               dim1 = int(line_spl[0])
               dim2 = int(line_spl[1])
               prt1 = int(line_spl[2])
               prt2 = int(line_spl[3])
               line = f.readline().split() # gamma point is the first line
               i = (prt1-1)*dim+dim1-1
               j = (prt2-1)*dim+dim2-1
               K[i,j] = float(line[3])
            #end if
            line = f.readline()
        #end if
    #end with
    return K
#end def

def load_force_constants_vasp(fname, num_prt, dim=3):
    import h5py
    f = h5py.File(fname,mode='r')

    K_raw = array(f['force_constants'])
    p2s   = array(f['p2s_map'])

    # this could probably be done more efficiently with array operations
    K = zeros((dim*num_prt,dim*num_prt))
    for prt1 in range(num_prt):
        for prt2 in range(num_prt):
           sprt2 = p2s[prt2]
           for dim1 in range(dim):
               for dim2 in range(dim):
                   i = prt1*dim + dim1
                   j = prt2*dim + dim2
                   K[i,j] = K_raw[prt1,sprt2,dim1,dim2]
               #end for
            #end for
        #end for
    #end for    
    f.close()

    # assume conversion from eV/Angstrom**2 to Ry/Bohr**2
    eV_A2 = 27.211399/2*0.529189379**-2
    return K/eV_A2
#end def



def load_phonon_modes(fname,num_prt,drop_modes=0):
    ws    = []
    vects = []

    with open(fname) as f:
        line = f.readline()
        while line:
            line_spl = line.split()
            if len(line_spl)==9 and line_spl[0]=='freq':
               freq = float(line_spl[7])
               vect = []
               for prt in range(num_prt):
                   line = f.readline().split() # gamma point is the first line
                   vect.append( float(line[1]) )
                   vect.append( float(line[3]) )
                   vect.append( float(line[5]) )
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

    if len(v.shape)==1: # if only one mode
        v = array([v])
    #end if
    return array(w),array(v)
#end def


# read geometry from qe format to row format
def read_geometry(geometry_string):
    lines = geometry_string.split('\n')
    R = []
    names = []
    for line in lines:
        fields = line.split()
        if len(fields)>3:
            names.append(fields[0])
            R.append(float(fields[1]))
            R.append(float(fields[2]))
            R.append(float(fields[3]))
       #end if
    #end for
    return array(R),names
#end def

# positions are given as a vector
def print_qe_geometry(atoms, positions,dim=3):
    for a,atom in enumerate(atoms):
        coords = ''
        for i in range(dim*a,dim*a+dim):
            coords += str(positions[i]).ljust(20)
        print(atom.ljust(6)+coords)
    #end for
#end def


def print_fc_matrix(fc, num_prt, diagonals_only=True, title=None):
    if not title==None:
        print('Force-constant matrix: '+title)
    #end if
     
    dim = 3
    for i1,prt1 in enumerate(range(num_prt)):
        for i2,prt2 in enumerate(range(num_prt)):
            for r,xyz in enumerate(['x','y','z',]):
                repr_str = str(xyz)+str(i1)+' '+str(xyz)+str(i2)
                ph_str   = str(fc[i1*dim+r,i2*dim+r])
                print(repr_str.ljust(16)+ph_str.ljust(16))
            #end for
        #end for
    #end for
    print('')
#end def

# w must be 1-d array of frequencies in cm-1
# vects must be 2-d array: num_freq x num_prt*dim
# masses is 1-d vector of num_prt
# there may be a better way to do this, but does it matter
def K_from_W(w,v,masses,dim=3):
    w2 = (deepcopy(array(w))*2*0.000004556335281)**2 # cm-1 to rydberg
    sqrt_m = sqrt(diag(repeat(masses,dim)))   # multiply by sqrt of masses
    # correctly normalize disp vectors
    v2 = deepcopy(v) @ sqrt_m
    for row in range(v2.shape[0]):
        v2[row,:] *= 1/linalg.norm(v2[row,:])
    #end for
    K = sqrt_m @ transpose(v2) @ diag(w2) @ v2 @ sqrt_m
    return K
#end def

# courtesy of Jaron Krogel
# XYp = x**0 y**0, x**0 y**1, x**0 y**2, ...
def bipolynomials(X,Y,nx,ny):
    X = X.flatten()
    Y = Y.flatten()
    Xp = [0*X+1.0]
    Yp = [0*Y+1.0]
    for n in range(1,nx+1):
        Xp.append(X**n)
    #end for
    for n in range(1,ny+1):
        Yp.append(Y**n)
    #end for
    XYp = []
    for Xn in Xp:
        for Yn in Yp:
            XYp.append(Xn*Yn)
        #end for
    #end for
    return XYp
#end def bipolynomials

# courtesy of Jaron Krogel
def bipolyfit(X,Y,Z,nx,ny):
    XYp = bipolynomials(X,Y,nx,ny)
    p,r,rank,s = linalg.lstsq(array(XYp).T,Z.flatten())
    return p
#end def bipolyfit


# courtesy of Jaron Krogel
def bipolyval(p,X,Y,nx,ny):
    shape = X.shape
    XYp = bipolynomials(X,Y,nx,ny)
    Z = 0*X.flatten()
    for pn,XYn in zip(p,XYp):
        Z += pn*XYn
    #end for
    Z.shape = shape
    return Z
#end def bipolyval

# courtesy of Jaron Krogel
def bipolymin(p,X,Y,nx,ny,itermax=6,shrink=0.1,npoints=10):
    for i in range(itermax):
        Z = bipolyval(p,X,Y,nx,ny)
        X=X.ravel()
        Y=Y.ravel()
        Z=Z.ravel()
        imin = Z.argmin()
        xmin = X[imin]
        ymin = Y[imin]
        zmin = Z[imin]
        dx = shrink*(X.max()-X.min())
        dy = shrink*(Y.max()-Y.min())
        xi = linspace(xmin-dx/2,xmin+dx/2,npoints)
        yi = linspace(ymin-dy/2,ymin+dy/2,npoints)
        X,Y = meshgrid(xi,yi)
        X = X.T
        Y = Y.T
    #end for
    return xmin,ymin,zmin
#end def bipolymin

def get_1d_sli(p,slicing):
    sli = deepcopy(slicing)
    if len(slicing)==1:
        sli[p] = (slice(None,None,None))
    elif p==0:
        sli[1] = (slice(None,None,None))
    elif p==1:
        sli[0] = (slice(None,None,None))
    else:
        sli[p] = (slice(None,None,None))
    #end if
    return tuple(sli)
#end def

def get_2d_sli(p0,p1,slicing):
    sli = deepcopy(slicing)
    if p0==0 and p1>1:
        sli[1]  = (slice(None,None,None))
        sli[p1] = (slice(None,None,None))
    elif p0==1 and p1>1:
        sli[0]  = (slice(None,None,None))
        sli[p1] = (slice(None,None,None))
    else:
        sli[p0] = (slice(None,None,None))
        sli[p1] = (slice(None,None,None))
    #end if
    return tuple(sli)
#end def

def get_min_params(shifts,PES,n=2,endpts=[]):
    pf     = polyfit(shifts,PES,n)
    r      = roots(polyder(pf))
    Pmins  = list(r[r.imag==0].real)
    for pt in endpts:
        Pmins.append(pt)
    #end for
    Emins = polyval(pf,array(Pmins))
    try:
        imin = argmin(Emins)
        Emin = Emins[imin]
        Pmin = Pmins[imin]
    except:
        Pmin = nan
        Emin = nan
    return Emin,Pmin,pf
#end def


def W_to_R(W,H):
    R = (2*W/H)**0.5
    return R
#end def

def R_to_W(R,H):
    W = 0.5*H*R**2
    return W
#end def

def get_fraction_error(data,fraction):
    data   = array(data)
    data   = data[~isnan(data)]        # remove nan
    ave    = mean(data)
    data   = data[data.argsort()]-ave  # sort and center
    pleft  = abs(data[int(len(data)*fraction)])
    pright = abs(data[int(len(data)*(1-fraction))])
    err    = max(pleft,pright)
    return ave,err
#end def


def merge_pos_cell(pos,cell):
    posc = array(list(pos.flatten())+list(cell.flatten())) # generalized position vector: pos + cell
    return posc
#end def

# assume 3x3 
def detach_pos_cell(posc,num_prt=None,dim=3,reshape=True):
    posc = posc.reshape(-1,dim)
    if num_prt is None:
        pos  = posc[:-dim].flatten()
        cell = posc[-dim:].flatten()
    else:
        pos  = posc[:num_prt].flatten()
        cell = posc[num_prt:].flatten()
    #end if
    if reshape:
        return pos.reshape(-1,3),cell.reshape(-1,3)
    else:
        return pos,cell
    #end if
#end def


def get_relax_structure(
    path,
    suffix     = 'relax.in',
    pos_units  = 'B',
    relax_cell = False,
    dim        = 3,
    ):
    relax_path = '{}/{}'.format(path,suffix)
    try:
        from nexus import PwscfAnalyzer
        relax_analyzer = PwscfAnalyzer(relax_path)
        relax_analyzer.analyze()
    except:
        print('No relax geometry available: run relaxation first!')
    #end try

    # get the last structure
    eq_structure = relax_analyzer.structures[len(relax_analyzer.structures)-1]
    pos_relax    = eq_structure.positions.flatten()
    if relax_cell:
        cell_relax = eq_structure.axes.flatten()
        pos_relax = array(list(pos_relax)+list(cell_relax)) # generalized position vector: pos + cell
    #end if
    return pos_relax
#end def


def compute_hessian(jax_hessian,eps=0.001,**kwargs):
    if jax_hessian:
        print('Computing parameter Hessian with JAX')
        hessian_delta = compute_hessian_jax(**kwargs)
    else:
        print('Computing parameter Hessian with finite difference')
        hessian_delta = compute_hessian_fdiff(eps=eps,**kwargs)
    #end if
    return hessian_delta
#end def


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
    for i,r in enumerate(pos):
        def pp(p):
            return params_to_pos(p)[i]
        #end def
        gradf = grad(pp)
        gradfs.append( gradf(p0) )
    #end for
    gradfs = array(gradfs)
    hessian_delta = gradfs.T @ hessian_pos @ gradfs
    savetxt('H_delta.dat',hessian_delta)
    return hessian_delta
#end def


def compute_jacobian_fdiff(params_to_pos,params,eps=0.001):
    jacobian = []
    pos_orig = params_to_pos(params)
    for p in range(len(params)):
        p_this     = params.copy()
        p_this[p] += eps
        pos_this   = params_to_pos(p_this)
        jacobian.append( (pos_this-pos_orig)/eps )
    #end for
    jacobian = array(jacobian).T
    return jacobian
#end def


# finite difference method for the gradient
def compute_hessian_fdiff(
    hessian_pos,
    params_to_pos,
    pos_to_params,
    pos,
    eps  = 0.001,
    **kwargs 
    ):
    params   = pos_to_params(pos)
    jacobian = compute_jacobian_fdiff(params_to_pos,params=params,eps=eps)
    hessian_delta = jacobian.T @ hessian_pos @ jacobian
    return hessian_delta
#end try


def print_hessian_delta(hessian_delta,U,Lambda,roundi=3):
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

def print_relax(elem,pos_relax,params_relax,dim=3):
    print('Relaxed geometry (non-symmetrized):')
    print_qe_geometry(elem,pos_relax,dim)
    print('Parameter values (non-symmetrized):')
    for p,pval in enumerate(params_relax):
        print(' #{}: {}'.format(p,pval))
    #end for
#end def



def calculate_X_matrix(x_n,pfn):
   X = []
   for x in x_n:
       row = []
       for pf in range(pfn+1):
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

def model_statistical_bias(pf,x_n,sigma):
    pfn = len(pf)-1
    # opposite index convention 
    p   = flipud(pf)

    X = calculate_X_matrix(x_n,pfn)
    F = calculate_F_matrix(X)
    s2 = sigma**2*(F @ F.T)

    if pfn==2:
        bias = (s2[1,2]/p[2]**2 - s2[2,2]*p[1]/p[2]**3)/2
    elif pfn==3:
        z = (p[2]**2-3*p[1]*p[3])**0.5
        b11 = s2[1,1]*(-3/4*p[3]*z**-3)/2
        b22 = s2[2,2]*(1/3/p[3]/z-p[2]**2/3/p[3]/z**3)/2
        b33 = s2[3,3]*(1/3/p[3])*(2/p[3]**2*(z-p[2]) + 3*p[1]/p[3]/z - 9/4*p[1]**2/z**3)/2
        b12 = s2[1,2]*(p[2]/z**3)/2
        b13 = s2[1,3]*(-3/2*p[1]/z**3)/2
        b23 = s2[2,3]*(p[1]*p[2]/p[3]/z**3 - 2*(p[2]/z-1)/3/p[3]**2 )/2
        bias = b11 + b22 + b33 + b12 + b13 + b23
    else: # no correction known
        bias = 0.0
    #end if
    return bias
#end def


def plot_U_heatmap(ax,U,sort=True,cmap='RdBu',labels=True):
    U_cp  = U.copy()
    D     = U.shape[0]
    ticks = range(D)
    if sort:
        U_sorted    = []
        yticklabels = []
        rows        = list(range(D))
        for d in range(D):
            # find the largest abs value of d from the remaining rows
            i = abs(U_cp[rows[d:],d]).argmax()
            rows = rows[:d] + [rows.pop(d+i)] + rows[d:]
        #end for
        U_sorted = array(U_cp[rows])
        yticklabels = rows
    else:
        U_sorted    = U_cp
        yticklabels = [ str(d) for d in range(D) ]
    #end if
    cb = ax.imshow(U_sorted,cmap=cmap,vmin=-1.0,vmax=1.0,aspect='equal')
    ax.set_yticks(ticks)
    ax.set_xticks(ticks)
    ax.set_yticklabels(yticklabels)
    if labels:
        ax.set_ylabel('Directions')
        ax.set_xlabel('Parameters')
    #end if
    return cb
#end def
