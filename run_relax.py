#! /usr/bin/env python3

from parameters import *

# here for the time being
settings(**srg_settings)
relax = get_srg_job()[0]

if __name__=='__main__':
    run_project(relax)
#end if

try:
    relax_analyzer = relax.load_analyzer_image()
    eq_structure   = relax_analyzer.structures[len(relax_analyzer.structures)-1]
    eq_pos         = reshape(eq_structure.positions,shp1)
except:
    print('No relax geometry available: run relaxation first!')
#end try

