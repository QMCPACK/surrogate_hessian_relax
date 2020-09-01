#! /usr/bin/env python3

from parameters import *
from nexus import settings,run_project
from surrogate import IterationData,surrogate_diagnostics

from run_phonon import FC_param as hessian
from run_phonon import R_relax  as pos_init

ls_settings = obj(
    get_jobs      = get_jobs
    pos_to_params = pos_to_params,
    pfn           = 4,
    S             = 7,
    epsilon       = 0.01,
    )
data_ls = [] # first iteration

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
data_ls.append(data)

# second iteration
data = data.iterate(ls_settings=ls_settings,divide_epsilon=2)
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
data_ls.append(data)

# third iteration, etc

if __name__=='__main__':
    surrogate_diagnostics(data_ls)
#end if
