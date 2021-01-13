#!/usr/bin/env python3

import pickle
from numpy import array,loadtxt,zeros,dot,diag,transpose,sqrt,repeat,linalg,reshape,meshgrid,poly1d,polyfit,polyval,argmin,linspace,random,ceil,diagonal,amax,argmax,pi,isnan,nan,mean,var,amin,isscalar,roots,polyder
from copy import deepcopy
from scipy.interpolate import interp1d

from surrogate_tools import W_to_R,R_to_W,get_min_params,get_fraction_error


# class for line-search iteration

class IterationData():

    def __init__(
        self,
        get_jobs      = None,               # function handle to get nexus jobs
        n             = 0,                  # iteration number
        path          = '../ls',            # directory
        # parameters, hessian and mappings
        params        = None,               # starting parameters (overrides pos if in conflict)
        pos           = None,               # starting generalized positions
        hessian       = None,               # parameter hessian
        params_to_pos = None,               # mapping from params to pos
        pos_to_params = None,               # mapping from pos to params
        # line-search properties
        pfn           = 3,                  # polyfit degree
        pts           = 7,                  # points for fit
        windows       = None,               # list of windows for each direction
        W             = 0.01,               #   alternative: constant energy window
        noises        = None,               # list of target noises for each direction
        add_noise     = 0.0,                #   additionally: constant artificial noise
        generate      = 1000,               # how many samples to generate for error analysis
        fraction      = 0.025,              # fraction for error analysis
        # calculation method specifics
        type          = 'qmc',              # job type: qmc/scf
        qmc_idx       = 1,                  #   qmc: which qmc to analyze
        qmc_j_idx     = 2,                  #   qmc: which qmc job has jastrow
        load_postfix  = '/dmc/dmc.in.xml',  #   qmc: point the input file
        # misc properties
        eqm_str       = 'eqm',              # eqm string
        targets       = None,               # list of parameter targets, if known
        colors        = None,               # list of colors for parameters and directions, otherwise randomize
        ):

        self.get_jobs      = get_jobs
        self.n             = n
        self.path          = path+'/ls'+str(n)+'/'  # format for line-search directions

        self.pfn           = pfn
        self.pts           = pts
        self.windows       = windows
        self.W             = W
        self.noises        = noises
        self.add_noise     = add_noise
        self.generate      = generate
        self.fraction      = fraction

        self.type          = type
        self.qmc_idx       = qmc_idx
        self.qmc_j_idx     = qmc_j_idx if type=='qmc' else 0
        self.load_postfix  = load_postfix

        self.eqm_str       = eqm_str
        self.targets       = targets
        self.colors        = colors

        # initiate variables
        self.eqm_path        = self.path+self.eqm_str
        self.is_noisy        = ( type=='qmc' or add_noise>0 or not noises is None )
        self.results_loaded  = False
        self.mappings_loaded = False
        self.hessian_loaded  = False

        # if provided, initiate params and mappings
        if not params_to_pos is None and not pos_to_params is None:
            if not params is None:
                self.load_params_mappings(params, params_to_pos, pos_to_params)
            elif not pos is None:
                self.load_pos_mappings(pos, params_to_pos, pos_to_params)
            #end if
            if not hessian is None:
                self.load_hessian(hessian)
            #end if
        #end if
    #end def

    # load starting params and mappings
    def load_params_mappings(self, params, params_to_pos, pos_to_params):
        self.params          = params
        self.pos             = params_to_pos(params)
        self.P               = len(self.params)
        self.params_to_pos   = params_to_pos
        self.pos_to_params   = pos_to_params
        self.mappings_loaded = True
        # backward compatibility
        self.F    = params_to_pos
        self.Finv = pos_to_params
    #end def

    # load starting pos and mappings
    def load_pos_mappings(self, pos, params_to_pos, pos_to_params):
        params = pos_to_params(pos)
        self.load_params_mappings(params, params_to_pos, pos_to_params)
    #end def

    # load parameter Hessian
    def load_hessian(self, hessian):
        Lambda,U        = linalg.eig(hessian)
        self.U          = U.T # linalg.eig has opposite convention; here U is D x P
        self.Lambda     = abs(Lambda)
        self.D          = len(Lambda)
        self.hessian    = hessian
        # windows
        if self.windows is None:
            self.windows = array(self.D*[self.W])
        #end if
        if self.noises is None:
            self.noises  = array(self.D*[self.add_noise])
        #end if
        self.hessian_loaded = True
    #end def

    # creates geometries according to shifts in optimal directions
    def shift_positions(self,D_list=None):
        if D_list is None:
            D_list = range(self.D) # by default, shift all
        #end if
        shift_data    = []
        shifts        = []

        sigma_min     = 1.0e99
        for d in range(self.D):
            if not d in D_list:
                shift_data.append([])
                shifts.append([])
            #end if
            shifts_d     = self._shift_parameter(d)
            sigma        = self.noises[d]
            minuss       = len(shifts_d[shifts_d<-1e-10])
            pluss        = 1
            shift_rows   = []
            # assume that shifts come in order
            for s,shift in enumerate(shifts_d):
                if abs(shift)<1e-10: #eqm
                    path = self.eqm_path
                elif shift<0.0:
                    path = self.path+'d'+str(d)+'_m'+str(minuss)
                    minuss -= 1
                else:
                    path = self.path+'d'+str(d)+'_p'+str(pluss)
                    pluss += 1
                #end if
                pos = self.pos.copy() + self._shift_position(d,shift)
                row = pos,path,sigma,shift
                shift_rows.append(row)
            #end for
            shift_data.append(shift_rows)
            shifts.append(shifts_d)
            sigma_min = min(sigma_min,sigma)
        #end for
        self.sigma_min  = sigma_min
        self.shifts     = shifts
        self.shift_data = shift_data # dimensions: P x S x (pos,path,sigma,shift)
    #end def

    # requires that nexus has been initiated
    def get_job_list(self):
        # eqm jobs
        eqm_jobs = self.get_jobs(self.pos,path=self.eqm_path,sigma=self.sigma_min)
        jobs     = eqm_jobs
        for d in range(self.D):
            for s in range(self.pts):
                pos,path,sigma,shift = self.shift_data[d][s]
                if not path==self.eqm_path:
                    if self.type=='qmc':
                        jastrow_job = eqm_jobs[self.qmc_j_idx]
                        jobs += self.get_jobs(pos,path=path,sigma=sigma,jastrow=jastrow_job)
                    else:
                        jobs += self.get_jobs(pos,path=path,sigma=sigma)
                    #end if
                #end if
            #end for
        #end for
        return jobs
    #end def

    # load the result of parallel line-search: energies, shifted positions, etc
    def load_results(self):
        # load eqm
        E,Err,kappa = self._load_energy_error(self.eqm_path+self.load_postfix,self.sigma_min)
        self.E     = E
        self.Err   = Err
        self.kappa = kappa
        # load ls
        Epred          = 1.0e99
        kappa_max      = kappa
        PES            = []
        PES_err        = []
        Dshifts        = []
        for d in range(self.D):
            PES_row     = []
            PES_err_row = []
            shifts      = []
            for s in range(self.pts):
                pos,path,sigma,shift = self.shift_data[d][s]
                if path==self.eqm_path:
                    E    = self.E
                    Err  = self.Err
                else:
                    E,Err,kappa = self._load_energy_error(path+self.load_postfix,sigma)
                #end if
                if E < Epred:
                    Epred     = E
                    Epred_err = Err
                #end if
                shifts.append(shift)
                PES_row.append(E)
                PES_err_row.append(Err)
            #end for
            Dshifts.append(shifts)
            PES.append(PES_row)
            PES_err.append(PES_err_row)
        #end for
        self.PES       = array(PES)
        self.PES_err   = array(PES_err)
        self.Epred     = Epred
        self.Epred_err = Epred_err
        self.Dshifts   = Dshifts

        self._compute_next_params()
        self._compute_next_hessian()
        self._compute_next_pos()
        self.results_loaded = True
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
        ls_settings    = dict(), # or nexus obj
        ):

        # backwards compatibility
        try:
            self.results_loaded = self.ready
        except:
            if not self.results_loaded:
                print('New position not calculated. Returning None')
                return None
            #end if
        #end try
        n = self.n+1 # advance iteration

        new_data = IterationData(n=n, pos=self.pos_next, hessian=self.hessian, **ls_settings)
        new_data.shift_positions()
        return new_data
    #end def


    # takes optimized line-search parameters from another IterationData instance
    def copy_optimized_parameters(self, data):
        self.pfn      = data.pfn
        self.pts      = data.pts
        # TODO: make this more robust
        if data.type=='scf' and self.type=='qmc':
            self.noises   = array(data.noises)/2 # from Ry to Ha
            self.windows  = data.windows
        else:
            self.noises   = data.noises
            self.windows  = data.windows
        #end if
        self.is_noisy = True
    #end def


    def _shift_parameter(self,d):
        pfn    = self.pfn
        pts    = self.pts
        H      = self.Lambda[d]
        W      = self.windows[d]
        R      = W_to_R(W,H)
        shifts = linspace(-R,R,pts)
        return shifts
    #end def


    def _shift_position(self,d,shift):
        dparams = self.params.copy() + self.U.T[:,d]*shift
        dpos    = self.params_to_pos(dparams)-self.params_to_pos(self.params)
        return dpos
    #end def


    def _load_energy_error(self,path,sigma):
        if self.type=='qmc':
            from nexus import QmcpackAnalyzer
            AI    = QmcpackAnalyzer(path)
            AI.analyze()
            E     = AI.qmc[self.qmc_idx].scalars.LocalEnergy.mean
            Err   = AI.qmc[self.qmc_idx].scalars.LocalEnergy.error
            kappa = AI.qmc[self.qmc_idx].scalars.LocalEnergy.kappa
        else: # pwscf
            from nexus import PwscfAnalyzer
            AI    = PwscfAnalyzer(path)
            AI.analyze()
            E     = AI.E + sigma*random.randn(1)[0]
            Err   = sigma
            kappa = 1.0
        #end if
        return E,Err,kappa
    #end def


    def _compute_next_hessian(self):
        Lambda_next = []
        for d in range(self.D):
            Lambda_next.append( self.pfs[d][-3] ) # pick the quadratic term
        #end for
        self.hessian_next = self.U.T @ diag(Lambda_next) @ self.U
        self.Lambda_next = Lambda_next
    #end def
     

    def _compute_next_params(self):
        if self.is_noisy:
            Emins     = []
            Emins_err = []
            Dmins     = []
            Dmins_err = []
            pfs       = []
            Ds        = []
            for d in range(self.D):
                Emins_blk = []
                Dmins_blk = []
                pfs_blk   = []
                Gs        = random.randn(self.generate,self.pts)
                PES       = self.PES[d]
                PES_err   = self.PES_err[d]
                shifts    = self.Dshifts[d]
                PES_fit   = polyval(polyfit(shifts,PES,self.pfn),shifts)
                for G in Gs:
                    PES_row = PES_fit + PES_err*G
                    Emin,Dmin,pf = get_min_params(shifts,PES_row,self.pfn)
                    Emins_blk.append(Emin)
                    Dmins_blk.append(Dmin)
                    pfs_blk.append(pf)
                #end for
                pf_ave        = mean(pfs_blk,axis=0)
                Emin,Emin_err = get_fraction_error(Emins_blk, self.fraction)
                Dmin,Dmin_err = get_fraction_error(Dmins_blk, self.fraction)
                Dmins.append(Dmin)
                Dmins_err.append(Dmin_err)
                Emins.append(Emin)
                Emins_err.append(Emin_err)
                pfs.append(pf_ave)
                Ds.append(Dmins_blk)
            #end for

            # propagate search error
            Ps = self.params + array(Ds).T @ self.U
            P_aves = []
            P_errs = []
            for p in range(self.P):
                P_ave,P_err = get_fraction_error( Ps[:,p], self.fraction )
                P_aves.append( Ps ) 
                P_errs.append( P_err ) 
            #end for
            self.params_next     = Ps[0]
            self.params_next_err = array(P_errs)
        else:
            Emins = []
            Dmins = []
            pfs   = []
            for d in range(self.D):
                Emin,Dmin,pf = get_min_params(self.Dshifts[d],self.PES[d],self.pfn)
                Emins.append(Emin)
                Dmins.append(Dmin)
                pfs.append(pf)
            #end for
            self.params_next     = self.params + self.U.T @ Dmins
            self.params_next_err = 0.0*self.params_next
            Dmins_err            = 0.0*array(Dmins)
            Emins_err            = 0.0*array(Emins)
        #end if
        self.Dmins     = array(Dmins)
        self.Emins     = array(Emins)
        self.Dmins_err = array(Dmins_err)
        self.Emins_err = array(Emins_err)
        self.pfs       = pfs
    #end def

    
    def _compute_next_pos(self):
        self.pos_next = self.params_to_pos(self.params_next)
    #end def


#end class

