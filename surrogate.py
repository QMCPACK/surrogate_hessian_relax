#!/usr/bin/env python3

import pickle
from numpy import array,loadtxt,zeros,dot,diag,transpose,sqrt,repeat,linalg,reshape,meshgrid,poly1d,polyfit,polyval,argmin,linspace,random,ceil,diagonal,amax,argmax,pi,isnan
from copy import deepcopy
from math import log10
from numerics import jackknife
from nexus import obj,PwscfAnalyzer,QmcpackAnalyzer
from matplotlib import pyplot as plt


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


def param_representation(Delta_p,K_cart):
    K_param = transpose(Delta_p) @ K_cart @ Delta_p
    return K_param
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

def get_min_params(shifts,PES,n=2):
    pf     = polyfit(shifts,PES,n)
    c      = poly1d(pf)
    crit   = c.deriv().r
    r_crit = crit[crit.imag==0].real
    test   = c.deriv(2)(r_crit)

    # compute local minima
    # excluding range boundaries
    # choose the one closest to zero
    min_idx = argmin(abs(r_crit[test>0]))
    Pmin    = r_crit[test>0][min_idx]
    Emin    = c(Pmin)
    return Emin,Pmin,pf
#end def



# Line-search related functions and classes

# a is the prefactor of bias scaling, depends on
#   force-constant k
#   anharmonicity in terms of depth (Morse-like) or a_idx, where a_idx<1 for most systems
def calculate_a(k,pfn,a_idx=1.0,De=None,pfnp1=0.0):
    # old rule of thumb: De**-3/2*k ~ 2*pi
    pfnp1 = abs(pfnp1)
    if not De is None: 
        if pfn==2:
            a = De**-0.5*k**-0.5*pi**2/18
        else:
            a = De**-1.5*k**-0.5/6/pi
        #end if
    elif pfnp1 > 0.0:
        if pfn==2:
            a = pfnp1
        elif pfn==3:
            a = pfnp1**1.5/2
        else:
            a = 2*pfnp1/3
        #end if
    else:
        if pfn==2: 
            a = k**-1.0*a_idx # to be updated
        else: # use rule of thumb: k**2/3*De**-3/2 ~ 2*pi
            #a = k**-1.5/3*a_idx
            a = k**(-7/6)/3*a_idx
        #end if
    #end if
    return a
#end def

# b is the prefactor of noise scaling
def calculate_b(k,pfn,S):
    if pfn==2:
        b = (S*k)**-0.5
    else:
        b = 3*(S*k)**-0.5
    #end if
    return b
#end def


def surrogate_anharmonicity(data,pfn=None,output=False):
    if pfn is None:
        pfn = data.pfn
    #end if
    W      = data.W
    pfnp1s = []
    ans    = []
    for p in range(data.P):
        shift = data.Dshifts[p]
        PES   = data.PES[p]
        H     = data.hessian_e[p]
        pf    = polyfit(shift,PES,pfn)
        pfnp1 = polyfit(shift,PES,pfn+1)
        dpf   = pfnp1[1:]-pf
        if pfn==2:
            an = -pfnp1[0]*H**-2*3/2
        elif pfn==3:
            #an = pfnp1[0]**1.5*H**-3.5 + dpf[1]/W
            an = pfnp1[0]**1.5*H**-3.5
        else: # n==4
            #an = 5/4*pfnp1[0]*H**-3
            an = 5/4*pfnp1[0]*H**-3
        #end if
        ans.append(an)
        if output:
            print('parameter #'+str(p)+', poly'+str(pfn)+', W='+str(W))
            print('pfn - pfnp1: '+str(dpf.round(4)))
            print('pfnp1[0]   : '+str(pfnp1[0].round(6)))
        #end if
    #end for
    ans        = array(ans)
    targets    = data.targets
    param_vals = data.param_vals_next
    print('D model    : '+str(ans*W**2))
    print('D bias     : '+str(data.Dmins))
    print('P model    : '+str(data.M @ ans *W**2))
    print('P bias     : '+str(param_vals-targets))
    print('anharm.    : '+str(ans))
    return ans
#end def



# a is the prefactor of bias scaling
#   depends on 
# b is the prefactor of sigma scaling
#   depends on
# epsilon is the target accuracy
# the optimal solutions depend on the order of the fit n
#
# returns: optimal energy window, target input errorbars
def get_optimal_noise_window(a, b, epsilon=0.01, pfn=3):
    if pfn==2:
        W_E      = epsilon/3/a
        sigma_in = 2*a/b*(epsilon/3/a)**1.5
    elif pfn==3 or pfn==4:
        W_E      = (epsilon/5/a)**0.5
        sigma_in = 4*a/b*(epsilon/5/a)**(5/4)
    else: 
        print('The order not supported')
    #end if
    return W_E, sigma_in
#end def


def print_structure_shift(pos_old,pos_new):
    print('New geometry:')
    print(R_new.reshape((-1,3)))
    print('Shift:')
    print((R_new-R_old).reshape((-1,3)))
#end for


def print_optimal_parameters(data_list):
    data0 = data_list[0]
    if data0.targets is None:
        target = array(data0.P*[0])
    else:
        target = data0.targets
    #end if
    Epred = None
    print('        Eqm energy     Predicted energy')
    for n in range(len(data_list)):
        E,Err          = data_list[n].E,data_list[n].Err
        if Epred is None:
            print('   n='+str(n)+': '+print_with_error(E,Err).ljust(15))
        else:
            print('   n='+str(n)+': '+print_with_error(E,Err).ljust(15)+print_with_error(Epred,Eprederr).ljust(15))
        #end if
        Epred,Eprederr = data_list[n].Epred,data_list[n].Epred_err
    #end for
    print('   n='+str(n+1)+': '+' '.ljust(15)+print_with_error(Epred,Eprederr).ljust(15))

    print('Optimal parameters:')
    for p in range(data0.P):
        print(' p'+str(p) )
        PV_this = data_list[0].param_vals[p] # first value
        print('  init: '+str(PV_this.round(8)).ljust(12))
        for n in range(len(data_list)):
            PV_this = data_list[n].param_vals[p]
            PV_next = data_list[n].param_vals_next[p]
            PV_err  = data_list[n].param_vals_next_err[p]
            print('   n='+str(n)+': '+print_with_error(PV_next,PV_err).ljust(12)+' Delta: '+print_with_error(PV_next-PV_this,PV_err).ljust(12))
        #end for
        if not target is None:
            print('  targ: '+str(target[p]).ljust(12))
        #end if
    #end for
#end def


def plot_energy_convergence(
        ax,
        data_list,
        target     = 0.0,
        linestyle  = ':',
        marker     = 'x',
        color      = 'b',
        pcolor     = 'r',
        pmarker    = 'v',
        show_pred  = True,
        label      = 'E (eqm)',
        plabel     = 'E (pred)',
        ):
    Es         = []
    Errs       = []
    Epreds     = []
    Eprederrs  = []
    for data in data_list:
        Es.append(data.E - target)
        Errs.append(data.Err)
        Epreds.append(data.Epred)
        Eprederrs.append(data.Epred_err)
    #end for
    ax.errorbar(range(len(data_list)),    
                Es,    
                Errs,
                linestyle  = linestyle,
                color      = color,
                marker     = marker, 
                label      = label,
                )
    if show_pred:
        ax.errorbar(range(1,len(data_list)+1),
                    Epreds,
                    Eprederrs,
                    linestyle = linestyle,
                    color     = pcolor,
                    marker    = pmarker,
                    label     = plabel,
                    )
    #end if
    ax.legend()
    ax.set_title('Equilibrium energy vs iteration')
    ax.set_xlabel('iteration')
    ax.set_ylabel('energy')
#end def


def plot_parameter_convergence(
        ax,
        data_list,
        colors    = None,
        targets   = None,
        label     = '',
        marker    = 'x',
        linestyle = ':',
        uplims    = True,
        lolims    = True,
        **kwargs
        ):
    ax.set_xlabel('iteration')
    ax.set_ylabel('parameter')
    ax.set_title('Parameters vs iteration')

    data0 = data_list[0]
    if not targets is None:
        targets = targets
    elif not data0.targets is None:
        targets = data0.targets
    else:
        targets = data0.param_vals # no target, use initial values
    #end if
    if colors is None:
        colors = data0.colors
    else:
        colors = colors
    #end if

    # init values
    P_vals = []
    P_errs = []
    for p in range(data0.P):
        P_vals.append([data0.param_vals[p]-targets[p]])
        P_errs.append([0.0])
    #end for
    # line search params
    for data in data_list:
        for p in range(data.P):
            P_vals[p].append(data.param_vals_next[p]-targets[p])
            P_errs[p].append(data.param_vals_next_err[p])
        #end for
    #end for
    # plot
    for p in range(data0.P):
        P_val   = P_vals[p]
        P_err   = P_errs[p]
        co      = colors[p]
        P_label = 'p'+str(p)+' '+label
        h,c,f   = ax.errorbar(list(range(len(data_list)+1)),P_val,P_err,
            color     = co,
            marker    = marker,
            linestyle = linestyle,
            label     = P_label,
            uplim     = uplims,
            lolims    = lolims
            )
        c[0].set_marker('_')
        c[1].set_marker('_')
    #end for
    ax.plot([0,len(data_list)],[0,0],'k-')
    ax.legend()
#end def


def plot_linesearches(ax,data_list):
    n         = len(data_list)
    data0     = data_list[0]
    max_shift = amax(abs(array(data0.shifts)))
    colors    = data0.colors

    xtlabel = []
    xtval   = []
    labels  = []
    for n,data in enumerate(data_list):
        xoffset = n*max_shift
        xtval.append(xoffset)
        xtlabel.append(str(n))
        for s,shift in enumerate(data.shifts):
            PES       = data.PES[s]
            PES_err   = data.PES_err[s]
            pf        = data.pfs[s]
            Dmin      = data.Dmins[s]
            Emin      = data.Emins[s]
            Dmin_err  = data.Dmins_err[s]
            Emin_err  = data.Emins_err[s]
            co        = colors[s]
            # plot PES
            s_axis = linspace(min(shift),max(shift))
            # plot fitted PES
            if data.is_noisy:
                ax.errorbar(shift+xoffset,PES,PES_err,linestyle='None',color=co,marker='.')
                ax.errorbar(Dmin+xoffset,Emin,xerr=Dmin_err,yerr=Emin_err,marker='x',color=co)
            else:
                ax.plot(shift+xoffset,PES,linestyle='None',color=co,marker='.')
                ax.plot(Dmin+xoffset,Emin,marker='x',color=co)
            #end if
            ax.plot(s_axis+xoffset,polyval(pf,s_axis),linestyle=':',color=co,label='p'+str(s))
            ax.plot(2*[xoffset],[min(PES),max(PES)],'k-')
        #end for
    #end for
    ax.set_xticks(xtval)
    ax.set_xticklabels(xtlabel)
    ax.set_xlabel('shift along direction per iteration')
    ax.set_ylabel('energy')
#end def


def plot_error_cost(
        ax,
        data_list,
        p_idx     = 0,
        marker    = 'x',
        linestyle = ':',
        color     = 'b',
        target    = None,
        label     = '',
        max_error = True,
     ):
    costs  = []
    PVs    = []
    PVes   = []
    cost   = 0.0 # accumulated cost per iteration
    for data in data_list:
        PES       = array(data.PES)
        PES_error = array(data.PES_error)
        sh        = PES.shape
        for r in range(sh[0]):
            for c in range(sh[1]):
                cost += PES_error[r,c]**(-2)
            #end for
        #end for
        costs.append(cost)

        P_vals = data.P_vals_next[p_idx]
        P_errs = data.P_vals_err[p_idx]
        if target is None:
            PVs.append( abs(P_vals) + P_errs )
        else:
            PVs.append( abs(P_vals - target) + P_errs )
        #end if
        PVes.append( P_errs )
    #end for
    if max_error:
        ax.plot(
            costs,
            array(PVs)+array(PVes),
            color     = color,
            marker    = marker,
            linestyle = linestyle,
            label     = label,
            )
    else:
        ax.errorbar(
            costs,
            PVs,
            PVes,
            color     = color,
            marker    = marker,
            linestyle = linestyle,
            label     = label,
            )
    #end if
#end for


def plot_PES_fits(
        ax,
        data,
        datamarker    = '.',
        datalinestyle = 'None',
        minmarker     = 'o',
        **kwargs):

    for p in range(data.P):
        co        = data.colors[p]
        shifts    = data.Dshifts[p]
        PES       = data.PES[p]
        PES_err   = data.PES_err[p]
        pf        = data.pfs[p]
        Dmin      = data.Dmins[p]
        Emin      = data.Emins[p]
        Dmin_err  = data.Dmins_err[p]
        Emin_err  = data.Emins_err[p]

        # plot PES
        if data.is_noisy:
            label = 'Emin='+print_with_error(Emin,Emin_err)+' Dmin'+str(p)+'='+print_with_error(Dmin,Dmin_err)
            ax.errorbar(shifts,PES ,PES_err, color=co, marker=datamarker, linestyle=datalinestyle)
            ax.errorbar(Dmin  ,Emin,xerr=Dmin_err,yerr=Emin_err,color=co,label=label,linestyle='None',marker=minmarker)
        else:
            label = 'Emin='+str(Emin.round(6))+' Dmin'+str(p)+'='+str(Dmin.round(6))
            ax.plot(shifts,PES, color=co, marker=datamarker, linestyle=datalinestyle)
            ax.plot(Dmin  ,Emin,color=co,label=label,marker=minmarker,linestyle='None')
        #end if
        s_axis = linspace(min(shifts),max(shifts))
        ax.plot(s_axis,polyval(pf,s_axis),color=co,**kwargs)
    #end for
    ax.set_title('Line-search #'+str(data.n))
    ax.set_xlabel('shift along direction')
    ax.set_ylabel('energy')
    ax.legend(fontsize=8)
#end def



def surrogate_diagnostics(data_list):
    # print standard stuff
    #print_structure_shift(data.R,data.R_next)
    print_optimal_parameters(data_list)
    # plot energy convergence
#    f,ax = plt.subplots()
#    plot_energy_convergence(ax,data_list)
    # plot parameter convergence
    f,ax = plt.subplots()
    plot_parameter_convergence(ax,data_list)
    # plot line searches
    for data in data_list:
        f,ax = plt.subplots()
        plot_PES_fits(ax,data)
    #end for
    plt.show()
#end def


class IterationData():

    def __init__(
        self,
        pos_to_params = None,               # function handle to get parameters from pos
        get_jobs      = None,               # function handle to get nexus jobs
        n             = 0,                  # iteration number
        pfn           = 4,                  # polyfit degree
        S             = 7,                  # points for fit
        path          = '../ls',           # directory
        # line-search properties
        W             = 0.01,               # energy window
        anharmonicity = None,               # estimated anharmonicities
        epsilon       = 0.01,               # target accuracy for parameters
        use_optimal   = True,               # use optimal directions
        add_noise     = 0.0,                # add artificial noise
        generate      = 1000,               # generate samples
        # calculate method specifics
        type          = 'qmc',              # job type
        qmc_idx       = 1,
        qmc_j_idx     = 2,
        load_postfix  = '/dmc/dmc.in.xml',
        # misc properties
        eqm_str       = 'eqm',             # eqm string
        targets       = None,               # targets, if known
        colors        = None,               # colors for parameters
        ):

        self.pos_to_params = pos_to_params
        self.get_jobs      = get_jobs
        self.n             = n
        self.pfn           = pfn
        self.S             = S
        self.path          = path+'/ls'+str(n)+'/'  # format for line-search directions

        self.W             = W
        self.anharmonicity = anharmonicity
        self.epsilon       = epsilon
        self.use_optimal   = use_optimal
        self.add_noise     = add_noise
        self.generate      = generate

        self.type          = type
        self.qmc_idx       = qmc_idx
        self.qmc_j_idx     = qmc_j_idx if type=='qmc' else 0
        self.load_postfix  = load_postfix

        self.eqm_str       = eqm_str
        self.targets       = targets
        self.colors        = colors

        self.eqm_path      = self.path+self.eqm_str
        self.is_noisy      = ( type=='qmc' or add_noise>0 or not anharmonicity is None )
        self.ready         = False
    #end def

    def load_pos(self, pos):
        self.pos           = pos
        P,PV               = self.pos_to_params(pos)
        self.params        = P
        self.param_vals    = PV
        self.P             = len(PV)
    #end def

    def load_hessian(self, hessian): # takes the parameter Hessian
        eigs,vects      = linalg.eig(hessian)
        directions      = vects.T @ self.params
        M               = [] # propagation matrix from directions to params: dP = M x dD
        for d,direction in enumerate(directions):
            dP,dPV = self.pos_to_params(direction)
            M.append(dPV)
        #end for
        self.M          = array(M).T
        self.hessian_e  = eigs
        self.hessian_v  = vects.T
        self.directions = directions
        self.hessian    = hessian
        self.epsilond   = linalg.inv(abs(self.M)) @ array(self.P*[self.epsilon]).T # scale target accuracy
        print(self.epsilond)
    #end def


    def shift_positions(self,D_list=None):
        if D_list is None:
            D_list = range(self.P) # by default, shift all
        #end if
        shift_list    = []
        sigma_min     = 1.0e99
        for p in range(self.P):
            shifts,sigma = self._shift_parameter(p)
            direction    = self.directions[p]
            shift_rows   = []
            for s,shift in enumerate(shifts):
                if abs(shift)<1e-10: #eqm
                    path = self.eqm_path
                else:
                    path = self.path+'p'+str(p)+'_s'+str(s)
                #end if
                pos = self.pos.copy() + shift*direction
                row = pos,path,sigma,shift
                shift_rows.append(row)
            #end for
            shift_list.append(shift_rows)
            sigma_min = min(sigma_min,sigma)
        #end for
        self.sigma_min  = sigma_min
        self.shift_list = shift_list # dimensions: P x S x (pos,path,sigma,shift)
    #end def

    # requires that nexus has been initiated
    def get_job_list(self):
        # eqm jobs
        eqm_jobs = self.get_jobs(pos=self.pos,path=self.eqm_path,sigma=self.sigma_min)
        jobs     = eqm_jobs
        for p in range(self.P):
            for s in range(self.S):
                pos,path,sigma,shift = self.shift_list[p][s]
                if not path==self.eqm_path:
                    if self.type=='qmc':
                        jastrow_job = eqm_jobs[self.qmc_j_idx]
                        jobs += self.get_jobs(pos=pos,path=path,sigma=sigma,jastrow=jastrow_job)
                    else:
                        jobs += self.get_jobs(pos=pos,path=path,sigma=sigma)
                    #end if
                #end if
            #end for
        #end for
        return jobs
    #end def

    def load_results(self):
        # load eqm
        E_eqm,Err_eqm  = self._load_energy_error(self.eqm_path+self.load_postfix)
        sigma_eqm      = self.sigma_min
        self.E         = E_eqm # +sigma_eqm*random.randn(1)[0]
        self.Err       = Err_eqm+sigma_eqm
        # load ls
        Epred          = 1.0e99
        Emins          = []
        Dmins          = []
        Emins_err      = []
        Dmins_err      = []
        PES            = []
        PES_err        = []
        Dshifts        = []
        pfs            = []
        for p in range(self.P):
            PES_row     = []
            PES_err_row = []
            shifts      = []
            for s in range(self.S):
                pos,path,sigma,shift = self.shift_list[p][s]
                if path==self.eqm_path:
                    E   = self.E
                    Err = self.Err
                else:
                    E,Err = self._load_energy_error(path+self.load_postfix)
                    #E    += sigma*random.randn((1))[0]
                    Err  += sigma
                #end if
                if E < Epred:
                    Epred     = E
                    Epred_err = Err
                #end if
                shifts.append(shift)
                PES_row.append(E)
                PES_err_row.append(Err)
            #end for
            Emin,Emin_err,Dmin,Dmin_err,pf = self._get_shift_minimum(shifts,PES_row,PES_err_row)
            Emins.append(Emin)
            Dmins.append(Dmin)
            Emins_err.append(Emin_err)
            Dmins_err.append(Dmin_err)
            PES.append(PES_row)
            PES_err.append(PES_err_row)
            Dshifts.append(shifts)
            pfs.append(pf)
        #end for
        self.PES       = array(PES)
        self.PES_err   = array(PES_err)
        self.Epred     = Epred
        self.Epred_err = Epred_err
        self.Emins     = Emins
        self.Dmins     = Dmins
        self.Emins_err = Emins_err
        self.Dmins_err = Dmins_err
        self.Dshifts   = Dshifts
        self.pfs       = pfs

        self.ready = self._compute_next_pos()
    #end def

    def write_to_file(self):
        pickle.dump(self,open(self.path+'data.p',mode='wb'))
    #end def

    def load_from_file(self):
        try:
            data = pickle.load(open(self.path+'data.p',mode='rb'))
            return data
        except:
            return None
        #end try
    #end def

    # copies relevant information to new iteration
    def iterate(
            self, 
            ls_settings    = obj(),
            divide_epsilon = None, 
            divide_W       = None,
            ):

        if not self.ready:
            print('New position not calculated. Returning None')
            return None
        #end if
        n = self.n+1 # advance iteration

        if not divide_epsilon is None:
            epsilon = self.epsilon/divide_epsilon
            print('New target epsilon: '+str(epsilon))
            ls_settings.set(epsilon=epsilon)
        #end if
        if not divide_W is None:
            W = self.W/divide_W
            print('New target energy window: '+str(epsilon))
            ls_settings.set(W=W)
        #end if

        new_data = IterationData(n=n, **ls_settings)
        new_data.load_pos(self.pos_next)
        new_data.load_hessian(self.hessian)
        new_data.shift_positions()
        return new_data
    #end def


    def _shift_parameter(self,p):
        pfn = self.pfn
        S   = self.S
        k   = self.hessian_e[p]
        if self.anharmonicity is None: # use just fixed energy window
            W     = self.W
            sigma = self.add_noise
        else:
            a       = self.anharmonicity[p]
            b       = calculate_b(k=k,S=S,pfn=pfn)
            W,sigma = get_optimal_noise_window(abs(a),b,pfn=pfn,epsilon=abs(self.epsilond[p]))
            print(p,a,b,self.epsilond[p],W,sigma)
            print('bias: ' ,a*W**2)
            print('noise: ',sigma*b*W**-0.5)
            print('bias  (param):',abs(self.M[:,p])*a*W**2)
            print('noise (param):',sigma*b*abs(self.M[:,p])*W**-0.5)
        #end if
        lim    = (2*W/k)**0.5
        shifts = linspace(-lim,lim,S)
        return shifts,sigma
    #end def


    def _load_energy_error(self,path):
        if self.type=='qmc':
            AI  = QmcpackAnalyzer(path,equilibration=self.equilibration)
            AI.analyze()
            E   = AI.qmc[self.qmc_idx].scalars.LocalEnergy.mean
            Err = AI.qmc[self.qmc_idx].scalars.LocalEnergy.error
        else: # pwscf
            AI  = PwscfAnalyzer(path)
            AI.analyze()
            E   = AI.E
            Err = 0.0
        #end if
        return E,Err
    #end def
    
    def _get_shift_minimum(self,shifts,PES,PES_err):
        pfn = self.pfn
        if self.is_noisy:
            # generate random data for jackknife resampling
            generate = self.generate
            data = []
            for s,shift in enumerate(shifts):
                ste = PES_err[s]
                data.append( PES[s]+generate**0.5*ste*random.randn(generate) )
            #end for
            data = array(data).T
            # run jackknife
            jcapture = obj()
            jackknife(data     = data,
                      function = get_min_params,
                      args     = [shifts,None,pfn],
                      position = 1,
                      capture  = jcapture)
            Emin,Dmin,pf = jcapture.jmean
            Emin_err,Dmin_err,pf_err = jcapture.jerror
            print('jack:', Dmin)
            Emin,Dmin,pf = get_min_params(shifts,PES,pfn) # TODO: remove this when this is over
            print('acc:',  Dmin)
        else:
            Emin,Dmin,pf = get_min_params(shifts,PES,pfn)
            Emin_err,Dmin_err,pf_err = 0.,0.,0.
        #end if

        return Emin,Emin_err,Dmin,Dmin_err,pf
    #end def
    
    
    def _compute_next_pos(self):
        pos_next = self.pos.copy() + self.Dmins @ self.directions
        P,PV     = self.pos_to_params(pos_next)
        PV_err   = abs(self.M @ self.Dmins_err)
        self.pos_next            = pos_next
        self.param_vals_next     = PV
        self.param_vals_next_err = PV_err
        return True
    #end def

#end class

