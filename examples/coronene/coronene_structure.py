#! /usr/bin/env python3

from nexus import generate_physical_system,Structure
from numpy import array,diag,reshape,linalg
from surrogate import read_geometry

#settings for the structure
pos_str = '''
C       17.333479260  16.347254420  5.000000000
C       17.333479260  13.652745580  5.000000000
C       15.000000000  17.694435370  5.000000000
C       19.664315560  17.692928650  5.000000000
C       15.000000000  12.305564630  5.000000000
C       12.666520740  16.347254420  5.000000000
C       19.664315560  12.307071350  5.000000000
C       15.000000000  20.385896360  5.000000000
C       21.961941560  16.299061790  5.000000000
C       19.606008920  20.379716230  5.000000000
C       21.961941560  13.700938210  5.000000000
C       12.666520740  13.652745580  5.000000000
C       17.355987610  21.678675650  5.000000000
C       15.000000000   9.614103640  5.000000000
C       10.335684440  17.692928650  5.000000000
C       19.606008920   9.620283770  5.000000000
C       12.644012390  21.678675650  5.000000000
C       10.335684440  12.307071350  5.000000000
C       17.355987610   8.321324350  5.000000000
C       10.393991080  20.379716230  5.000000000
C       12.644012390   8.321324350  5.000000000
C        8.038058440  16.299061790  5.000000000
C       10.393991080   9.620283770  5.000000000
C        8.038058440  13.700938210  5.000000000
H       23.751175540  17.339410360  5.000000000
H       21.401536810  21.409185650  5.000000000
H       23.751175540  12.660589630  5.000000000
H       17.349782410  23.748412340  5.000000000
H       21.401536810   8.590814350  5.000000000
H       12.650217590  23.748412340  5.000000000
H       17.349782410   6.251587660  5.000000000
H        8.598463190  21.409185650  5.000000000
H       12.650217590   6.251587660  5.000000000
H        6.248824460  17.339410360  5.000000000
H        8.598463190   8.590814350  5.000000000
H        6.248824460  12.660589630  5.000000000
'''
dim        = 3
pos_init,elem = read_geometry(pos_str)
masses     = 24*[10947.356792250725] + 12*[918.68110941480279]
cell_init  = [30.0,30.0,10.0]
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

def pos_to_params(pos,cell=None):
    params = []
    pval = []
    pos2 = reshape(pos,shp2)
    # param 1: CC inside
    C03_00  = pos2[ 3,:]-pos2[ 0,:]
    C06_01  = pos2[ 6,:]-pos2[ 1,:]
    C13_04  = pos2[13,:]-pos2[ 4,:]
    C17_11  = pos2[17,:]-pos2[11,:]
    C14_05  = pos2[14,:]-pos2[ 5,:]
    C07_02  = pos2[ 7,:]-pos2[ 2,:]
    r1 = array([ C03_00       , # C0
                 C06_01       , # C1
                 C07_02       , # C2
                 C03_00       , # C3
                 C13_04       , # C4
                 C14_05       , # C5
                 C06_01       , # C6
                 C07_02       , # C7
                 C03_00       , # C8
                 C03_00       , # C9
                 C06_01       , # C10
                 C17_11       , # C11
                 C07_02       , # C12
                 C13_04       , # C13
                 C14_05       , # C14
                 C06_01       , # C15
                 C07_02       , # C16
                 C17_11       , # C17
                 C13_04       , # C18
                 C14_05       , # C19
                 C13_04       , # C20
                 C14_05       , # C21
                 C17_11       , # C22
                 C17_11       , # C23
                 C03_00       , # H0  /24
                 C03_00       , # H1  /25
                 C06_01       , # H2  /26
                 C07_02       , # H3  /27
                 C06_01       , # H4  /28
                 C07_02       , # H5  /29
                 C13_04       , # H6  /30
                 C14_05       , # H7  /31
                 C13_04       , # H8  /32
                 C14_05       , # H9  /33
                 C17_11       , # H10 /34
                 C17_11       , # H11 /35
                 ]).reshape(shp1)
    params.append( r1/linalg.norm(r1) )
    pval.append(linalg.norm(pos2[0,:]-pos2[1,:]))
    # param 2: CC semi-in
    zero    = array([0,0,0])
    r2 = array([ zero         , # C0
                 zero         , # C1
                 zero         , # C2
                 C03_00       , # C3
                 zero         , # C4
                 zero         , # C5
                 C06_01       , # C6
                 C07_02       , # C7
                 C03_00       , # C8
                 C03_00       , # C9
                 C06_01       , # C10
                 zero         , # C11
                 C07_02       , # C12
                 C13_04       , # C13
                 C14_05       , # C14
                 C06_01       , # C15
                 C07_02       , # C16
                 C17_11       , # C17
                 C13_04       , # C18
                 C14_05       , # C19
                 C13_04       , # C20
                 C14_05       , # C21
                 C17_11       , # C22
                 C17_11       , # C23
                 C03_00       , # H0  /24
                 C03_00       , # H1  /25
                 C06_01       , # H2  /26
                 C07_02       , # H3  /27
                 C06_01       , # H4  /28
                 C07_02       , # H5  /29
                 C13_04       , # H6  /30
                 C14_05       , # H7  /31
                 C13_04       , # H8  /32
                 C14_05       , # H9  /33
                 C17_11       , # H10 /34
                 C17_11       , # H11 /35
                 ]).reshape(shp1)
    params.append( r2/linalg.norm(r2) )
    pval.append(linalg.norm(C03_00))
    # param 3: CC semi-out
    C12_07  = pos2[12,:]-pos2[ 7,:]
    C09_03  = pos2[ 9,:]-pos2[ 3,:]
    C08_03  = pos2[ 8,:]-pos2[ 3,:]
    C10_06  = pos2[10,:]-pos2[ 6,:]
    C15_06  = pos2[15,:]-pos2[ 6,:]
    C18_13  = pos2[18,:]-pos2[13,:]
    C20_13  = pos2[20,:]-pos2[13,:]
    C22_17  = pos2[22,:]-pos2[17,:]
    C23_17  = pos2[23,:]-pos2[17,:]
    C21_14  = pos2[21,:]-pos2[14,:]
    C19_14  = pos2[19,:]-pos2[14,:]
    C16_07  = pos2[16,:]-pos2[ 7,:]
    r3 = array([ zero         , # C0
                 zero         , # C1
                 zero         , # C2
                 zero         , # C3
                 zero         , # C4
                 zero         , # C5
                 zero         , # C6
                 zero         , # C7
                 C08_03+C10_06, # C8
                 C09_03+C12_07, # C9
                 C08_03+C10_06, # C10
                 zero         , # C11
                 C09_03+C12_07, # C12
                 zero         , # C13
                 zero         , # C14
                 C15_06+C18_13, # C15
                 C16_07+C19_14, # C16
                 zero         , # C17
                 C15_06+C18_13, # C18
                 C16_07+C19_14, # C19
                 C22_17+C20_13, # C20
                 C23_17+C21_14, # C21
                 C22_17+C20_13, # C22
                 C23_17+C21_14, # C23
                 C08_03+C10_06, # H0  /24
                 C09_03+C12_07, # H1  /25
                 C08_03+C10_06, # H2  /26
                 C09_03+C12_07, # H3  /27
                 C15_06+C18_13, # H4  /28
                 C16_07+C19_14, # H5  /29
                 C15_06+C18_13, # H6  /30
                 C16_07+C19_14, # H7  /31
                 C22_17+C20_13, # H8  /32
                 C23_17+C21_14, # H9  /33
                 C22_17+C20_13, # H10 /34
                 C23_17+C21_14, # H11 /35
                 ]).reshape(shp1)
    params.append( r3/linalg.norm(r3) )
    pval.append(linalg.norm(pos2[9]-pos2[3]))
    # r4: CC outside
    C12_09  = pos2[12,:]-pos2[ 9,:]
    C08_10  = pos2[ 8,:]-pos2[10,:]
    C15_18  = pos2[15,:]-pos2[18,:]
    C20_22  = pos2[20,:]-pos2[22,:]
    C23_21  = pos2[23,:]-pos2[21,:]
    C19_16  = pos2[19,:]-pos2[16,:]
    r4 = array([ zero         , # C0
                 zero         , # C1
                 zero         , # C2
                 zero         , # C3
                 zero         , # C4
                 zero         , # C5
                 zero         , # C6
                 zero         , # C7
                 C08_10       , # C8
                 C12_09       , # C9
                -C08_10       , # C10
                 zero         , # C11
                -C12_09       , # C12
                 zero         , # C13
                 zero         , # C14
                 C15_18       , # C15
                -C19_16       , # C16
                 zero         , # C17
                -C15_18       , # C18
                 C19_16       , # C19
                 C20_22       , # C20
                -C23_21       , # C21
                -C20_22       , # C22
                 C23_21       , # C23
                 C08_10       , # H0  /24
                -C12_09       , # H1  /25
                -C08_10       , # H2  /26
                 C12_09       , # H3  /27
                 C15_18       , # H4  /28
                -C19_16       , # H5  /29
                -C15_18       , # H6  /30
                 C19_16       , # H7  /31
                 C20_22       , # H8  /32
                -C23_21       , # H9  /33
                -C20_22       , # H10 /34
                 C23_21       , # H11 /35
                 ]).reshape(shp1)
    params.append( r4/linalg.norm(r4) )
    pval.append(linalg.norm(pos2[12]-pos2[9]))
    # r5: CH
    C12_H27  = pos2[12,:]-pos2[27,:]
    C09_H25  = pos2[ 9,:]-pos2[25,:]
    C08_H24  = pos2[ 8,:]-pos2[24,:]
    C10_H26  = pos2[10,:]-pos2[26,:]
    C15_H28  = pos2[15,:]-pos2[28,:]
    C18_H30  = pos2[18,:]-pos2[30,:]
    C20_H32  = pos2[20,:]-pos2[32,:]
    C22_H34  = pos2[22,:]-pos2[34,:]
    C23_H35  = pos2[23,:]-pos2[35,:]
    C21_H33  = pos2[21,:]-pos2[33,:]
    C19_H31  = pos2[19,:]-pos2[31,:]
    C16_H29  = pos2[16,:]-pos2[29,:]
    r5 = array([ zero         , # C0
                 zero         , # C1
                 zero         , # C2
                 zero         , # C3
                 zero         , # C4
                 zero         , # C5
                 zero         , # C6
                 zero         , # C7
                 zero         , # C8
                 zero         , # C9
                 zero         , # C10
                 zero         , # C11
                 zero         , # C12
                 zero         , # C13
                 zero         , # C14
                 zero         , # C15
                 zero         , # C16
                 zero         , # C17
                 zero         , # C18
                 zero         , # C19
                 zero         , # C20
                 zero         , # C21
                 zero         , # C22
                 zero         , # C23
                 C08_H24      , # H0  /24
                 C09_H25      , # H1  /25
                 C10_H26      , # H2  /26
                 C12_H27      , # H3  /27
                 C15_H28      , # H4  /28
                 C16_H29      , # H5  /29
                 C18_H30      , # H6  /30
                 C19_H31      , # H7  /31
                 C20_H32      , # H8  /32
                 C21_H33      , # H9  /33
                 C22_H34      , # H10 /34
                 C23_H35      , # H11 /35
                 ]).reshape(shp1)
    params.append( r5/linalg.norm(r5) )
    pval.append(linalg.norm(pos2[24]-pos2[8]))
    # r6: HH
    H24_H26  = pos2[24,:]-pos2[26,:]
    H27_H25  = pos2[27,:]-pos2[25,:]
    H28_H30  = pos2[28,:]-pos2[30,:]
    H31_H29  = pos2[31,:]-pos2[29,:]
    H32_H34  = pos2[32,:]-pos2[34,:]
    H35_H33  = pos2[35,:]-pos2[33,:]
    r6 = array([ zero         , # C0
                 zero         , # C1
                 zero         , # C2
                 zero         , # C3
                 zero         , # C4
                 zero         , # C5
                 zero         , # C6
                 zero         , # C7
                 zero         , # C8
                 zero         , # C9
                 zero         , # C10
                 zero         , # C11
                 zero         , # C12
                 zero         , # C13
                 zero         , # C14
                 zero         , # C15
                 zero         , # C16
                 zero         , # C17
                 zero         , # C18
                 zero         , # C19
                 zero         , # C20
                 zero         , # C21
                 zero         , # C22
                 zero         , # C23
                 H24_H26      , # H0  /24
                -H27_H25      , # H1  /25
                -H24_H26      , # H2  /26
                 H27_H25      , # H3  /27
                 H28_H30      , # H4  /28
                -H31_H29      , # H5  /29
                -H28_H30      , # H6  /30
                 H31_H29      , # H7  /31
                 H32_H34      , # H8  /32
                -H35_H33      , # H9  /33
                -H32_H34      , # H10 /34
                 H35_H33      , # H11 /35
                 ]).reshape(shp1)
    params.append( r6/linalg.norm(r6) )
    pval.append(linalg.norm(pos2[31]-pos2[29]))   

    return array(params),array(pval)
#end def

