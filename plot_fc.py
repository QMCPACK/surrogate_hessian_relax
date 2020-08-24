#!/usr/bin/env python3

from parameters import *
from run_phonon import *
from matplotlib import pyplot as plt
from numpy import polyfit,linalg,linspace,random

# figure for PES derivatives
def plot_dpes(ax):
    for p in range(num_params):
        dE = PES_dEs[p].copy()
        dE-= min(dE)
        fc = FC_param[p,p]
        shift = S_orig[p]
        dx = linspace(min(shift),max(shift),101)
        co = random.random((3,))
        ax.plot(shift,dE,'x',color=co)
        ax.plot(dx,0.5*dx**2*fc,'-',color=co,label='p'+str(p))
        ax.set_ylabel('dE')
        ax.set_xlabel('dp'+str(p))
    #end for
    ax.legend()
    ax.set_title(label+' PES gradients')
#end def

def plot_PES_contour(p0,p1,ax,levels=20):
    X = S_orig_mesh[p0][get_2d_sli(p0,p1,slicing)]+P_val[p0]
    Y = S_orig_mesh[p1][get_2d_sli(p0,p1,slicing)]+P_val[p1]
    Z = PES_param[get_2d_sli(p0,p1,slicing)]
    ax.contourf(X,Y,Z,levels)

    pvects = FC_v[[[p0],[p1]],[p0,p1]]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    c   = ((xlim[1]-xlim[0])**2+(ylim[1]-ylim[0])**2)**0.5/8

    ax.arrow( P_val[p0], P_val[p1], c*pvects[0,0], c*pvects[1,0], width=c/16 )
    ax.arrow( P_val[p0], P_val[p1], c*pvects[0,1], c*pvects[1,1], width=c/16 )

    ax.set_xlabel('p'+str(p0))
    ax.set_ylabel('p'+str(p1))
#end def

def plot_structure(ax,p,color_list=None):
    pos2 = reshape(R_relax,shp2)
    ax.set_xlim([0,a])
    ax.set_ylim([0,a])

    # plot eqm structure in xy projection
    elem_list  = list(set(elem))
    if color_list is None:
        color_list = list(random.random((len(elem_list),3)))
    #end if
    for prt in range(num_prt):
        color = color_list[elem_list.index(elem[prt])] # find color for particle
        ax.plot(pos2[prt,axes[0]],pos2[prt,axes[1]],'o',color=color)
    #end for

    # plot param displacements
    param2 = array([reshape(P_orig[p,:],shp2)[:,axes[0]],reshape(P_orig[p,:],shp2)[:,axes[1]]])
    # plot displacement
    for prt in range(num_prt):
        color = color_list[elem_list.index(elem[prt])] # find color for particle
        x = pos2[prt,axes[0]]
        y = pos2[prt,axes[1]]
        dx = param2[0,prt]
        dy = param2[1,prt]
        if abs(dx+dy)>1e-10:
            ax.arrow(x,y,dx,dy,head_width=0.4, head_length=0.2, fc=color, ec=color)
        #end if
    #end for
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_aspect('equal') # equal aspect
    ax.set_title(label+': original parameter p'+str(p))
#end def

color_elem = list(random.random((len(list(set(elem))),3)))
for p0 in range(num_params):
    f,ax = plt.subplots()
    plot_structure(ax,p0,color_elem)
#end for

plt.show()

