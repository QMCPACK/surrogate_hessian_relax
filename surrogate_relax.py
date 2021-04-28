#!/usr/bin/env python3

from numpy import array,loadtxt,zeros,dot,diag,transpose,sqrt,repeat,linalg,reshape,meshgrid,poly1d,polyfit,polyval,argmin,linspace,random,ceil,diagonal,amax,argmax,pi,isnan,nan,mean,var,amin,sum
from matplotlib import pyplot as plt

from surrogate_tools import print_with_error,model_statistical_bias
from iterationdata import IterationData,W_to_R,R_to_W


# Utility for plotting energy convergence (needs update)
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

# Utility for plotting parameter convergence
def plot_parameter_convergence(
        ax,
        data_list,
        P_list    = None,
        colors    = None,
        targets   = None,
        label     = '',
        markers   = None,
        marker    = 'x',
        linestyle = ':',
        uplims    = True,
        lolims    = True,
        labels    = None,
        offset    = 0,
        **kwargs
        ):
    ax.set_xlabel('iteration')
    ax.set_ylabel('parameter')
    ax.set_title('Parameters vs iteration')

    data0 = data_list[0]
    if P_list is None:
        P_list = range(data0.P)
    #end if
    if not targets is None:
        targets = targets
    elif not data0.targets is None:
        targets = data0.targets
    else:
        targets = data0.params # no target, use initial values
    #end if
    if colors is None:
        colors = data0.colors
    else:
        colors = colors
    #end if
    if markers is None:
        markers = data0.P*[marker]
    #end if

    # init values
    P_vals = []
    P_errs = []
    for p in range(data0.P):
        P_vals.append([data0.params[p]-targets[p]+offset])
        P_errs.append([0.0])
    #end for
    # line search params
    for data in data_list:
        for p in range(data0.P):
            P_vals[p].append(data.params_next[p]-targets[p]+offset)
            P_errs[p].append(data.params_next_err[p])
        #end for
    #end for
    # plot
    for p in range(data0.P):
        P_val   = P_vals[p]
        P_err   = P_errs[p]
        co      = colors[p]
        mk      = markers[p]
        if labels is None:
            P_label = 'p'+str(p)+' '+label
        else:
            P_label = labels[p]
        #end if
        if p in P_list:
            h,c,f   = ax.errorbar(list(range(len(data_list)+1)),P_val,P_err,
                color     = co,
                marker    = mk,
                linestyle = linestyle,
                label     = P_label,
                uplims    = uplims,
                lolims    = lolims,
                **kwargs,
                )
            if uplims or lolims:
                c[0].set_marker('_')
                c[1].set_marker('_')
            #end if
        #end if
    #end for
    ax.set_xticks(range(len(data_list)+1))
    ax.plot([0,len(data_list)],2*[offset],'k-')
    #ax.legend()
#end def

# Deprecated
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

# Plot parameter errors and accumulated cost (deprecated)
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
    data0 = data_list[0]
    if target is None:
        if data0.targets is None:
            target = 0.0
        else:
            target = data0.targets[p_idx]
        #end if
    #end if
    costs  = []
    PVs    = []
    PVes   = []
    cost   = 0.0 # accumulated cost per iteration
    for data in data_list:
        PES       = array(data.PES)
        PES_err   = array(data.PES_err)
        sh        = PES.shape
        for r in range(sh[0]):
            for c in range(sh[1]):
                cost += PES_err[r,c]**(-2)
            #end for
        #end for
        costs.append(cost)

        P_vals = data.params_next[p_idx]
        P_errs = data.params_next_err[p_idx]
        PVs.append( abs(P_vals - target) + P_errs )
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

# Plot the PES of each line-search in a parallel iteration
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

# Centralized set of actions to run as diagnostics
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

# Print energies and result parameters (could use a face-lift)
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
        PV_this = data_list[0].params[p] # first value
        print('  init: '+str(PV_this.round(8)).ljust(12))
        for n in range(len(data_list)):
            PV_this = data_list[n].params[p]
            PV_next = data_list[n].params_next[p]
            PV_err  = data_list[n].params_next_err[p]
            print('   n='+str(n)+': '+print_with_error(PV_next,PV_err).ljust(12)+' Delta: '+print_with_error(PV_next-PV_this,PV_err).ljust(12))
        #end for
        if not target is None:
            print('  targ: '+str(target[p]).ljust(12))
        #end if
    #end for
#end def

# Estimate average parameters and errors based on series of line-searches
#   transient=0 means all iterations from the first step forward will be included
# in future, set bias_correction=True by default. For now, set false to maintain integrity
def average_params(data_list,transient=0,bias_correction=False):
    params = []
    errs   = []
    for data in data_list[transient:]:
        param     = data.params_next.copy()
        param_err = data.params_next_err.copy()
        if bias_correction and data.is_noisy:
            param -= statistical_biases(data) @ data.U
        #end if
        params.append(param)
        errs.append(param_err)
    #end def
    params_ave = mean(array(params),axis=0)
    errs_ave   = sum(array(errs)**2,axis=0)**0.5/len(data_list[transient:])
    return params_ave,errs_ave
#end def

# Calculate statistical biases per direction based on model
def statistical_biases(data):
    S_biases = []
    for d in range(data.D):
        pf    = data.pfs[d]
        sigma = data.noises[d]
        x_n   = data.shifts[d]
        sbias = model_statistical_bias(pf,x_n,sigma)
        S_biases.append(sbias)
    #end for
    return array(S_biases)
#end def
