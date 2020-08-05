#!/usr/bin/env python3

from numpy import array,loadtxt,zeros,dot,diag,transpose,sqrt,repeat,linalg,reshape,meshgrid,poly1d,polyfit,polyval,argmin,linspace,random,ceil,diagonal
from copy import deepcopy
from numerics import jackknife
from nexus import obj,PwscfAnalyzer,QmcpackAnalyzer
from pickle import load,dump

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
    print('Total energy:')
    for n in range(len(data_list)):
       E,Err = data_list[n].E,data_list[n].Err
       print('   n='+str(n)+': '+print_with_error(E,Err)) 
    #end for
    print('Optimal parameters:')
    for p in range(data_list[0].disp_num):
        print(' p'+str(p) )
        PV_this = data_list[0].P_vals[p] # first value
        print('  init: '+str(PV_this))
        for n in range(len(data_list)):
            PV_this = data_list[n].P_vals[p]
            PV_next = data_list[n].P_vals_next[p]
            PV_err  = data_list[n].P_vals_err[p]
            print('   n='+str(n)+': '+print_with_error(PV_next,PV_err).ljust(12)+' Delta: '+print_with_error(PV_next-PV_this,PV_err).ljust(12))
        #end for
    #end for
#end def


from math import log10
def print_with_error( value, error, limit=15 ):

    if error==0.0:
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


class IterationData():

    def __init__(
        self,
        get_jobs,  # function handle to get nexus jobs
        n             = 0,
        E_lim         = 0.01,
        S_num         =  7,
        polyfit_n     = 4,
        dmcsteps      = 100,
        use_optimal   = True,
        path          = '../ls0/',
        type          = 'qmc',
        qmc_idx       = 1,
        qmc_j_idx     = 2,
        eqm_str       = 'eqm',
        equilibration = 10,
        load_postfix  = '/dmc/dmc.in.xml',
        noise         = 0.0,
        generate      = 1000,
        extras        = [],
        ):

        self.get_jobs      = get_jobs
        self.n             = n
        self.E_lim         = E_lim
        self.S_num         = S_num
        self.polyfit_n     = polyfit_n
        self.dmcsteps      = dmcsteps
        self.use_optimal   = use_optimal
        self.path          = path
        self.type          = type
        self.qmc_idx       = qmc_idx
        self.qmc_j_idx     = qmc_j_idx if type=='qmc' else 0
        self.eqm_str       = eqm_str
        self.equilibration = equilibration
        self.load_postfix  = load_postfix
        self.noise         = noise
        self.generate      = generate
        self.extras        = extras
        self.eqm_path      = self.path+self.eqm_str
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
        S_nums      = []
        for p in range(self.disp_num):
            lim = (2*self.E_lim/self.P_lims[p])**0.5
            shifts = list(linspace(-lim,lim,self.S_num))
            shifts_list.append(shifts)
            S_nums.append(len(shifts))
        #end for
        # add extras
        for p,extra in enumerate(self.extras):
            for shift in extra:
                shifts_list[p].append(shift)
                S_nums[p] += 1
            #end for
        #end for
        self.shifts = shifts_list
        self.S_nums = S_nums
    #end def
    
    # define paths and displacements
    def shift_structure(self):
        ls_paths = []
        R_shifts = []
        for p in range(len(self.shifts)):
            paths   = []
            R_shift = []
            disp    = self.disp[p,:]
            for s,shift in enumerate(self.shifts[p]):
                if abs(shift)<1e-10: #eqm
                    paths.append( self.eqm_path )
                else:
                    paths.append( self.path+'p'+str(p)+'_s'+str(s) )
                #end if
                R_shift.append( deepcopy(self.R) + shift*disp )
            #end for
            R_shifts.append(R_shift)
            ls_paths.append(paths)
        #end for
        self.R_shifts = R_shifts
        self.ls_paths = ls_paths
    #end def

    def load_PES(self):
        # load eqm
        E_eqm,Err_eqm  = self.load_energy_error(self.eqm_path+self.load_postfix)
        self.E         = E_eqm+self.noise*random.randn(1)[0]
        self.Err       = Err_eqm+self.noise
        self.PES       = []
        self.PES_error = []
        Epred          = 0.0
        # load ls
        for s,paths in enumerate(self.ls_paths):
            E_load   = []
            Err_load = []
            for path in paths:
                if path==self.eqm_path:
                    E_load.append(self.E)
                    Err_load.append(self.Err)
                else:
                    E,Err = self.load_energy_error(path+self.load_postfix)
                    E_load.append(E+self.noise*random.randn(1)[0])
                    Err_load.append(Err+self.noise)
                #end if
            #end for
            Epred = min(Epred,min(array(E_load)))
            self.PES.append(array(E_load))
            self.PES_error.append(array(Err_load))
        #end for
        self.Epred = Epred
        self.get_dp_E_mins()
        self.compute_new_structure()
    #end def

    def load_energy_error(self,path):
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
    

    def get_dp_E_mins(self):
        dPVs      = []
        dPVs_err  = []
        Emins     = []
        Emins_err = []
        pfs       = []
        pfs_err   = []
        for s,shift in enumerate(self.shifts):
            Emin,dPV,pf = get_min_params(shift,self.PES[s],n=self.polyfit_n)

            # generate random data
            data = []
            for d,dp in enumerate(shift):
                ste = self.PES_error[s][d]
                data.append( self.PES[s][d]+self.generate**0.5*ste*random.randn(self.generate) )
            #end for
            data = array(data).T
            # run jackknife
            jcapture = obj()
            jackknife(data     = data,
                      function = get_min_params,
                      args     = [shift,None,self.polyfit_n],
                      position = 1,
                      capture  = jcapture)
            Emin_err,dPV_err,pf_err = jcapture.jerror

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
        PV_err = self.P_vals*0
        for p,dPV in enumerate(self.dPV):
            R_next += dPV*self.disp[p,:]
            PV_err += self.pos_to_params(self.disp[p,:])[1]*self.dPV_err
        #end for
        P,PV_next        = self.pos_to_params(R_next)
        self.R_next      = R_next
        self.P_vals_next = PV_next
        self.P_vals_err  = PV_err
    #end def

    def plot_PES_fits(self,ax):
        for s,shift in enumerate(self.shifts):
            PES       = self.PES[s]
            PESe      = self.PES_error[s]
            pf        = self.pfs[s]
            Pmin      = self.dPV[s]
            Emin      = self.Emins[s]
            Pmin_err  = self.dPV_err[s]
            Emin_err  = self.Emins_err[s]
    
            # plot PES
            co = random.random((3,))
            s_axis = linspace(min(shift),max(shift))
            # plot fitted PES
            if self.type=='qmc' or self.noise>0.0:
                ax.errorbar(shift,PES,PESe,linestyle='None',color=co,marker='.')
                ax.errorbar(Pmin,Emin,xerr=Pmin_err,yerr=Emin_err,marker='o',color=co,
                            label='E='+print_with_error(Emin,Emin_err)+' dp'+str(s)+'='+print_with_error(Pmin,Pmin_err))
            else:
                ax.plot(shift,PES,linestyle='None',color=co,marker='.')
                ax.plot(Pmin,Emin,marker='o',color=co,
                            label='E='+str(Emin.round(6))+' dp'+str(s)+'='+str(Pmin.round(6)))
            #end if
            ax.plot(s_axis,polyval(pf,s_axis),linestyle=':',color=co)
            # plot minima
        #end for
        ax.set_title('Line-search #'+str(self.n))
        ax.set_xlabel('dp')
        ax.set_ylabel('E')
        ax.legend(fontsize=8)
    #end def

    def write_to_file(self):
        fname = self.path+'data.p'
        with open(fname,mode='wb') as f:
            dump(self,f)
        #end with
    #end def

#end class


