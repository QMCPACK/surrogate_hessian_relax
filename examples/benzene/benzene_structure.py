#! /usr/bin/env python3

from nexus import generate_physical_system,Structure
from numpy import array,diag,reshape,linalg,sin,cos,pi
from surrogate import read_geometry

#settings for the structure
a = 15.0
cell_init  = [a,a,a]
pos_init  = array([ 0.        ,  2.65075664,  0.        ,
                    0.        ,  4.70596609,  0.        ,
                   -2.29562041,  1.32537832,  0.        ,
                   -4.07549676,  2.35299249,  0.        ,
                   -2.29562041, -1.32537832,  0.        ,
                   -4.07549676, -2.3529925 ,  0.        ,
                    0.        , -2.65075664,  0.        ,
                    0.        , -4.70596609,  0.        ,
                    2.29562041, -1.32537832,  0.        ,
                    4.07549676, -2.3529925 ,  0.        ,
                    2.29562041,  1.32537832,  0.        ,
                    4.07549676,  2.35299249,  0.        ])+a/2
dim       = 3
elem      = 6*['C','H']
masses    = 6*[10947.356792250725,918.68110941480279]
relax_cell = False
num_prt    = len(elem)
shp2       = (num_prt+int(relax_cell)  ,dim)
shp1       = ((num_prt+int(relax_cell))*dim)

def generate_structure(pos_vect,cell_vect):
    structure = Structure(dim=dim)
    structure.set_axes(axes = diag(cell_vect))
    structure.set_elem(elem)
    structure.pos = reshape(pos_vect,shp2)
    structure.units = 'B'
    structure.add_kmesh(
        kgrid = (1,1,1), # Monkhorst-Pack grid
        kshift = (0,0,0) # and shift
    )
    return structure
#end def

def pos_to_params(pos):
    params = []
    pval   = []
    pos2 = pos.reshape(shp2)
    # param 1: CC distance (also shifts CH)
    r_CC = array([cos( 3*pi/6), sin( 3*pi/6), 0.,cos( 3*pi/6), sin(3*pi/6), 0.,
                  cos( 5*pi/6), sin( 5*pi/6), 0.,cos( 5*pi/6), sin(5*pi/6), 0.,
                  cos( 7*pi/6), sin( 7*pi/6), 0.,cos( 7*pi/6), sin(7*pi/6), 0.,
                  cos( 9*pi/6), sin( 9*pi/6), 0.,cos( 9*pi/6), sin(9*pi/6), 0.,
                  cos(11*pi/6), sin(11*pi/6), 0.,cos(11*pi/6),sin(11*pi/6), 0.,
                  cos(13*pi/6), sin(13*pi/6), 0.,cos(13*pi/6),sin(13*pi/6), 0.,])
    params.append( r_CC/linalg.norm(r_CC) )
    # param 2 : CH distance
    r_CH = array([ 0., 0., 0., cos( 3*pi/6), sin( 3*pi/6), 0.,
                   0., 0., 0., cos( 5*pi/6), sin( 5*pi/6), 0.,
                   0., 0., 0., cos( 7*pi/6), sin( 7*pi/6), 0.,
                   0., 0., 0., cos( 9*pi/6), sin( 9*pi/6), 0.,
                   0., 0., 0., cos(11*pi/6), sin(11*pi/6), 0.,
                   0., 0., 0., cos(13*pi/6), sin(13*pi/6), 0.,])
    params.append( r_CH/linalg.norm(r_CH) )
    pval = [abs(pos[1]-pos[19])/2,abs(pos[4]-pos[1])]
    return array(params),array(pval)
#end def

