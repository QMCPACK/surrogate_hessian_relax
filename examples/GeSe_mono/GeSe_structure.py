#! /usr/bin/env python3

from nexus import generate_physical_system,Structure
from numpy import array,diag,reshape,linalg
from surrogate import read_geometry

#settings for the structure
pos_str = '''
Ge   0.4000000000        0.250000000000000   0.5605
Se   0.500000000000000   0.250000000000000   0.4468
Se   0.000000000000000   0.750000000000000   0.5532
Ge   0.9000000000        0.750000000000000   0.4395
'''
pos_init,elem = read_geometry(pos_str)
dim           = 3
masses        = [1.0,1.0,1.0,1.0]
cell_init     = [8.32,7.35,40.780286834] 
relax_cell    = False
num_prt       = len(elem)
shp2          = (num_prt+int(relax_cell)  ,dim)
shp1          = ((num_prt+int(relax_cell))*dim)

def generate_structure(pos_vect,cell_diag):
    structure = Structure(dim=dim)
    structure.set_axes(axes = diag(cell_diag))
    structure.set_elem(elem)
    structure.pos = pos_vect.reshape(shp2)*cell_diag # note what happens here!
    structure.units = 'B'
    structure.add_kmesh(
        kgrid = (8,8,1), # Monkhorst-Pack grid
        kshift = (0,0,0) # and shift
    )
    return structure
#end def

def pos_to_params(pos): # pos given in crystal units
    params = []
    pval = []
    pos2 = reshape(pos,shp2)
    Ge1 = pos2[ 0,:]
    Se1 = pos2[ 1,:]
    Se2 = pos2[ 2,:]
    Ge2 = pos2[ 3,:]
    zero = [0,0,0]
    # x 
    x_p = array([ [1.0,0,0],
                  zero,
                  zero,
                  [1.0,0,0],
                ]).reshape(shp1)
    params.append( x_p/linalg.norm(x_p) )
    pval.append((Ge2[0]+Ge1[0]-0.5)/2)
    # z1
    z1_p = array([ [0,0,1.0],
                  zero,
                  zero,
                  [0,0,-1.0],
                ]).reshape(shp1)
    params.append( z1_p/linalg.norm(z1_p) )
    pval.append((Ge1[2]-Ge2[2]+1.0)/2)
    # z2
    z2_p = array([zero,
                  [0,0,-1.0],
                  [0,0,1.0],
                  zero
                ]).reshape(shp1)
    params.append( z2_p/linalg.norm(z2_p) )
    pval.append((Se2[2]-Se1[2]+1.0)/2)

    return array(params),array(pval)
#end def

def pos_to_params_cell(pos,a,b): # pos given in crystal units
    params = []
    pval = []
    pos2 = reshape(pos,shp2)
    Ge1 = pos2[ 0,:]
    Se1 = pos2[ 1,:]
    Se2 = pos2[ 2,:]
    Ge2 = pos2[ 3,:]
    zero = [0,0,0]
    # x
    x_p = array([ [1.0,0,0],
                  zero,
                  zero,
                  [1.0,0,0],
                ]).reshape(shp1)
    params.append( x_p/linalg.norm(x_p) )
    pval.append((Ge2[0]+Ge1[0]-0.5)/2)
    # z1
    z1_p = array([ [0,0,1.0],
                  zero,
                  zero,
                  [0,0,-1.0],
                ]).reshape(shp1)
    params.append( z1_p/linalg.norm(z1_p) )
    pval.append((Ge1[2]-Ge2[2]+1.0)/2)
    # z2
    z2_p = array([zero,
                  [0,0,-1.0],
                  [0,0,1.0],
                  zero
                ]).reshape(shp1)
    params.append( z2_p/linalg.norm(z2_p) )
    pval.append((Se2[2]-Se1[2]+1.0)/2)
    # a
    #params.append([])
    #pval.append(a)
    # b
    #pval.append(b)

    return array(params),array(pval)
#end def

