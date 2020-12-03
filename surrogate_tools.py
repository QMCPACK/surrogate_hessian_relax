#!/usr/bin/env python3

from numpy import array,loadtxt,zeros,dot,diag,transpose,sqrt,repeat,linalg,reshape,meshgrid,poly1d,polyfit,polyval,argmin,linspace,random,ceil,diagonal,amax,argmax,pi,isnan,nan,mean,var,amin,isscalar,roots,polyder,savetxt
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


def load_gamma_k(fname, num_prt, dim=3):
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


def compute_hessian_jax(hessian_real,params_to_pos,pos_to_params,pos_relax):
    from jax import grad
    print('Computing parameter Hessian with JAX')
    p0 = pos_to_params(pos_relax)
    gradfs = []
    for i,r in enumerate(pos_relax):
        def pp(p):
            return params_to_pos(p)[i]
        #end def
        gradf = grad(pp)
        gradfs.append( gradf(p0) )
    #end for
    gradfs = array(gradfs)
    hessian_delta = gradfs.T @ hessian_real @ gradfs
    savetxt('H_delta.dat',hessian_delta)
    return hessian_delta
#end def

def compute_hessian_fdiff(hessian_real,params_to_pos,pos_to_params,pos_relax):
    # finite difference method for the gradient
    print('Computing parameter Hessian with finite difference')
    eps = 0.002
    p0 = pos_to_params(pos_relax)
    gradfs = []
    for p in range(len(p0)):
        p_this     = p0.copy()
        p_this[p] += eps
        posi       = params_to_pos(p_this)
        gradfs.append( (posi-pos_relax)/eps )
    #end for
    gradfs = array(gradfs).T
    hessian_delta = gradfs.T @ hessian_real @ gradfs
    savetxt('H_delta.dat',hessian_delta)
    return hessian_delta
#end try

def print_hessian_delta(hessian_delta,U,Lambda,roundi=3):
    print('Parameters Hessian (H_Delta)')
    print(hessian_delta.round(roundi))
    print('')
    print('Eigenvectors (U):')
    print(U.round(roundi))
    print('')
    print('Eigenvalues (Lambda):')
    print(Lambda.round(roundi))
    print('')
#end def

def print_relax(elem,pos_relax,params_relax,cell_relax=None,dim=3):
    print('Relaxed geometry (non-symmetrized):')
    print_qe_geometry(elem,pos_relax,dim)
    print('Parameter values (non-symmetrized):')
    for p,pval in enumerate(params_relax):
        print(' #{}: {}'.format(p,pval))
    #end for
    if not cell_relax is None:
        print('Relaxed cell:')
        print(cell_relax.reshape((-1,dim)))
    #end if
#end def
