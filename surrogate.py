#!/usr/bin/env python3

from numpy import array,loadtxt,zeros,dot,diag,transpose,sqrt,repeat,linalg,reshape,meshgrid,poly1d,polyfit,polyval,argmin,linspace,random
from copy import deepcopy
from numerics import jackknife
from nexus import obj
from qmcpack_analyzer import QmcpackAnalyzer

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

# gets a list of pshift arrays (one list item for each parameter)
def get_pshift_list(pshifts):
    num_params = len(pshifts)
    ar = array(meshgrid(*pshifts)).T.reshape((-1,num_params))
    return ar
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


def bipolyfit(X,Y,Z,nx,ny):
    XYp = bipolynomials(X,Y,nx,ny)
    p,r,rank,s = linalg.lstsq(array(XYp).T,Z.flatten())
    return p
#end def bipolyfit


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
    pf = polyfit(shifts,PES,n)
    c = poly1d(pf)
    crit = c.deriv().r
    r_crit = crit[crit.imag==0].real
    test = c.deriv(2)(r_crit)

    # compute local minima
    # excluding range boundaries
    # choose the one closest to zero
    min_idx = argmin(abs(r_crit[test>0]))
    Pmin = r_crit[test>0][min_idx]
    Emin = c(Pmin)
    return Emin,Pmin,pf
#end def


def print_structure_shift(R_old,R_new):
    print('New geometry:')
    print(R_new.reshape((-1,3)))
    print('Shift:')
    print((R_new-R_old).reshape((-1,3)))
#end for


def print_optimal_parameters(data_list):
    print('Optimal parameters:')
    print(' init:' )
    PV_this = data_list[0].P_vals
    for p in range(len(PV_this)):
        print('  #'+str(p)+': '+str(PV_this[p]))
    #end for
    for n in range(len(data_list)):
        print(' n='+str(n) )
        PV_this = data_list[n].P_vals
        PV_next = data_list[n].P_vals_next
        for p in range(len(PV_this)):
            print('  #'+str(p)+': '+str(PV_next[p])+' Delta: '+str(PV_next[p]-PV_this[p]))
        #end for
    #end for
#end def


class IterationData():

    def __init__(
        self,
        n             = 0,
        E_lim         = 0.01,
        S_num         =  7,
        polyfit_n     = 4,
        dmc_factor    = 1,
        prefix        = '',
        qmc_idx       = 1,
        use_optimal   = True,
        path          = '../ls0/',
        eqm_str       = 'eqm',
        equilibration = 10,
        load_postfix  = '/dmc/dmc.in.xml',
        generate      = 1000,
        ):

        self.n             = n
        self.E_lim         = E_lim
        self.S_num         = S_num
        self.polyfit_n     = polyfit_n
        self.dmc_factor    = dmc_factor
        self.use_optimal   = use_optimal
        self.path          = path
        self.eqm_str       = eqm_str
        self.prefix        = prefix
        self.eqm_path      = self.path+self.eqm_str
        self.equilibration = equilibration
        self.load_postfix  = load_postfix
        self.generate      = generate
        self.qmc_idx       = qmc_idx
    #end def

    def load_R(self, R, func_params):
        self.R = R
        self.pos_to_params = func_params
        P,PV = self.pos_to_params(R)
        self.P_vals = PV
    #end def

    def load_displacements(self, P, P_lims):
        self.disp     = P
        self.disp_num = P.shape[0]
        self.P_lims   = P_lims
        self.set_optimal_shifts()
        self.shift_structure()
    #end def

    def set_optimal_shifts(self):
        shifts_list = []
        for p in range(self.disp_num):
            lim = (2*self.E_lim/self.P_lims[p])**0.5
            shifts = linspace(-lim,lim,self.S_num)
            shifts_list.append(shifts)
        #end for
        self.shifts = shifts_list
    #end def
    
    # define paths and displacements
    def shift_structure(self):
        ls_paths = []
        R_shift = []
        for p in range(len(self.shifts)):
            disp = self.disp[p,:]
            for s,shift in enumerate(self.shifts[p]):
                if abs(shift)<1e-10: #eqm
                    eqm_path = self.path+self.eqm_str
                    ls_paths.append( self.eqm_path )
                else:
                    ls_paths.append( self.path+self.prefix+'p'+str(p)+'_s'+str(s) )
                #end if
                R_shift.append( deepcopy(self.R) + shift*disp )
            #end for
        #end for
        self.R_shift  = R_shift
        self.eqm_path = eqm_path
        self.ls_paths = ls_paths
    #end def

    def load_PES(self):
        E_load   = []
        Err_load = []
        # load eqm
        E_eqm,Err_eqm = self.load_energy_error(self.eqm_path+self.load_postfix)
        self.E   = E_eqm
        self.Err = Err_eqm
        # load ls
        for s,path in enumerate(self.ls_paths):
            E,Err = self.load_energy_error(path+self.load_postfix)
            E_load.append(E)
            Err_load.append(Err)
        #end for
        self.PES       = array(E_load).reshape((self.disp_num,self.S_num))
        self.PES_error = array(Err_load).reshape((self.disp_num,self.S_num))

        self.get_dp_E_mins()
        self.compute_new_structure()
    #end def

    def load_energy_error(self,path):
        AI = QmcpackAnalyzer(path,equilibration=self.equilibration)
        AI.analyze()
        E   = AI.qmc[self.qmc_idx].scalars.LocalEnergy.mean
        Err = AI.qmc[self.qmc_idx].scalars.LocalEnergy.error
        return E,Err
    #end def
    
    
    def get_dp_E_mins(self):
        dPVs      = []
        dPVs_err  = []
        Emins     = []
        Emins_err = []
        pfs       = []
        pfs_err   = []
        for s,shift in enumerate(self.shifts):
            Emin,dPV,pf = get_min_params(shift,self.PES[s,:],n=self.polyfit_n)

            # generate random data
            data = []
            for d,dp in enumerate(shift):
                ste = self.PES_error[s,d]
                data.append( self.PES[s,d]+self.generate**0.5*ste*random.randn(self.generate) )
            #end for
            data = array(data).T
            # run jackknife
            jcapture = obj()
            jackknife(data = data,
                      function = get_min_params,
                      args     = [shift,None,self.polyfit_n],
                      position = 1,
                      capture  = jcapture)
            dPV_err,Emin_err,pf_err = jcapture.jerror

            dPVs.append(dPV)
            dPVs_err.append(dPV_err)
            Emins.append(Emin)
            Emins_err.append(Emin_err)
            pfs.append(pf)
            pfs_err.append(pf_err)
        #end for
        self.dPV        = dPVs
        self.dPV_err    = dPVs_err
        self.Emins      = Emins
        self.Emins_err  = Emins_err
        self.pfs        = pfs
        self.pfs_err    = pfs_err
    #end def
    
    
    def compute_new_structure(self):
        R_next = deepcopy(self.R)
        for p,dPV in enumerate(self.dPV):
            R_next += dPV*self.disp[p,:]
        #end for
        P,PV_next        = self.pos_to_params(R_next)
        self.R_next      = R_next
        self.P_vals_next = PV_next
    #end def

    def plot_PES_fits(self,ax):
        for s,shift in enumerate(self.shifts):
            PES       = self.PES[s,:]
            PESe      = self.PES_error[s,:]
            pf        = self.pfs[s]
            Pmin      = self.dPV[s]
            Emin      = self.Emins[s]
            Pmin_err  = self.dPV_err[s]
            Emin_err  = self.Emins_err[s]
    
            # plot PES
            co = random.random((3,))
            s_axis = linspace(min(shift),max(shift))
            # plot fitted PES
            ax.errorbar(shift,PES,PESe,linestyle='-',label='p'+str(s),color=co)
            ax.plot(s_axis,polyval(pf,s_axis),linestyle=':',color=co)
            # plot minima
            ax.errorbar(Pmin,Emin,yerr=Pmin_err,xerr=Emin_err,marker='o',color=co,
                        label='E='+print_with_error(Emin,Emin_err)+' p='+print_with_error(Pmin,Pmin_err))
        #end for
        ax.set_title('Line-search #'+str(self.n))
        ax.set_xlabel('dp')
        ax.set_ylabel('E')
        ax.legend()
    #end def

#end class


from math import log10
def print_with_error( value, error, limit=15 ):

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
