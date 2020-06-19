#! /usr/bin/env python3

from parameters import *

# here for the time being
settings(**relax_settings)
relax = get_relax_job()[0]

if __name__=='__main__':
    run_project(relax)
#end if

try:
    relax_analyzer = relax.load_analyzer_image()
    eq_structure   = relax_analyzer.structures[len(relax_analyzer.structures)-1]
    R_relax        = reshape(eq_structure.positions,shp1)
    if __name__=='__main__':
        print('Relaxed geometry:')
        print(R_relax.reshape(shp2))
    #end if
except:
    print('No relax geometry available: run relaxation first!')
#end try
