#!/usr/bin/env python3


from numpy import array, sin, cos, pi, exp

from surrogate_tools import mean_distances, bond_angle, distance

harmonic_a = lambda p,a: p[1]*(a-p[0])**2
# from Nexus
morse = lambda p,r: p[2]*((1-exp(-(r-p[0])/p[1]))**2-1)+p[3]

# test H2 molecule
pos_H2 = array('''
0.00000        0.00000        0.7
0.00000        0.00000       -0.7
'''.split(),dtype=float).reshape(-1,3)
elem_H2 = 'H H'.split()
def forward_H2(pos):
    r = distance(pos[0], pos[1])
    return [r]
#end def
def backward_H2(params):
    H1 = params[0]*array([0.0, 0.0, 0.5])
    H2 = params[0]*array([0.0, 0.0,-0.5])
    return array([H1, H2])
#end def
hessian_H2 = array([[1.0]])
def pes_H2(params):  # inaccurate; for testing
    r, = tuple(params)
    V = 0.0
    V += morse([1.4, 1.17, 0.5, 0.0], r)
    return V
#end def
def alt_pes_H2(params):  # inaccurate; for testing
    r, = tuple(params)
    V = 0.0
    V += morse([1.35, 1.17, 0.6, 0.0], r)
    return V
#end def


# test H2O molecule
pos_H2O = array('''
0.00000        0.00000        0.11779
0.00000        0.75545       -0.47116
0.00000       -0.75545       -0.47116
'''.split(),dtype=float).reshape(-1,3)
elem_H2O = 'O H H'.split()
def forward_H2O(pos):
    r_OH = mean_distances([(pos[0], pos[1]), (pos[0], pos[2])])
    a_HOH = bond_angle(pos[1], pos[0], pos[2])
    return [r_OH, a_HOH]
#end def
def backward_H2O(params):
    r_OH = params[0]
    a_HOH = params[1]*pi/180
    O = [0., 0., 0.]
    H1 = params[0]*array([0.0, cos((pi-a_HOH)/2), sin((pi-a_HOH)/2)])
    H2 = params[0]*array([0.0,-cos((pi-a_HOH)/2), sin((pi-a_HOH)/2)])
    return array([O, H1, H2])
#end def
hessian_H2O = array([[1.0, 0.2],
                     [0.2, 0.5]])  # random guess for testing purposes
def pes_H2O(params):
    r, a = tuple(params)
    V = 0.0
    V += morse([0.95789707, 0.5, 0.5, 0.0], r)
    V += harmonic_a([104.119, 0.5], a)
    return V
#end def


# test GeSe monolayer
#                    a     b     x       z1       z2
params_GeSe = array([4.26, 3.95, 0.4140, 0.55600, 0.56000])
elem_GeSe = 'Ge Ge Se Se'.split()
def forward_GeSe(pos, axes):
    Ge1, Ge2, Se1, Se2 = tuple(pos.split())
    a = axes[0,0]
    b = axes[1,1]
    x = mean([Ge1[0], Ge2[0] - 0.5])
    z1 = mean([Ge1[2], 1 - Ge2[2]])
    z2 = mean([Se2[2], 1 - Se1[2]])
    return [a, b, x, z1, z2]
#end def
def backward_GeSe(params):
    a, b, x, z1, z2 = tuple(params.split())
    Ge1 = [x,       0.25, z1]
    Ge2 = [x + 0.5, 0.75, 1 - z1]
    Se1 = [0.5,     0.25, 1 - z2]
    Se2 = [0.0,     0.75, z2]
    axes = diag([a, b, 20.0])
    pos = array([Ge1, Ge2, Se1, Se2])
    return pos, axes
#end def
# random guess for testing purposes
hessian_GeSe = array([[1.0,  0.5,  40.0,  50.0,  60.0],
                      [0.5,  2.5,  20.0,  30.0,  10.0],
                      [40.0, 20.0, 70.0,  30.0,  10.0],
                      [50.0, 30.0, 30.0, 130.0,  90.0],
                      [60.0, 10.0, 10.0,  90.0, 210.0]])
