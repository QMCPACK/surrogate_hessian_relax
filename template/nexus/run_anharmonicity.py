#! /usr/bin/env python3

from parameters import *
from nexus import settings,run_project
from surrogate import IterationData,surrogate_anharmonicity

from run_phonon import FC_param as hessian
from run_phonon import R_relax as pos_init

ls_settings = obj(
    pos_to_params = pos_to_params,
    get_jobs      = get_scf_job,
    S             = 9,
    pfn           = 4,
    W             = 0.32,
    path          = '../anharmonicity',
    type          = 'scf',
    load_postfix  = '/scf.in',
    targets       = [2.639303916,2.07130711],
    colors        = ['r','b']
    )

# first iteration, try to load data
data = IterationData(n=0, **ls_settings)
data.load_pos(pos_init)
data.load_hessian(hessian)
data.shift_positions()
data_load = data.load_from_file()
if data_load is None:
    settings(**nx_settings)
    jobs = data.get_job_list()
    run_project(jobs)
    data.load_results()
    data.write_to_file()
else:
    data = data_load
#end if

if __name__=='__main__':
    surrogate_anharmonicity(data,output=True)
#end if
