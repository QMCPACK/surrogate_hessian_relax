#! /usr/bin/env python3

from nexus import settings,run_project,obj
from surrogate import IterationData,surrogate_diagnostics

from parameters import pos_to_params,params_to_pos,get_prime_jobs,nx_settings

from run_phonon import hessian
from run_phonon import pos_relax
from run_error_scan import data as srg_error

n_max = 3 # number of iterations
ls_settings = obj(
    get_jobs      = get_prime_jobs
    pos_to_params = pos_to_params,
    type          = 'scf',
    load_postfix  = '/scf.in',
    path          = 'scf/',
    pfn           = srg_data.pfn,
    pts           = srg_data.pts,
    epsilon       = srg_data.epsilon,
    targets       = pos_to_params(pos_relax)
    )
data_ls = [] # first iteration

# first iteration
data = IterationData(n=0, pos=pos_init, hessian=hessian, **ls_settings)
data_load = data.load_from_file()
if data_load is None:
    data.shift_positions()
    settings(**nx_settings)
    jobs = data.get_job_list()
    run_project(jobs)
    data.load_results()
    data.write_to_file()
else:
    data = data_load
#end if
data_ls.append(data)

# repeat to n_max iterations
for n in range(1,n_max):
    data = data.iterate(ls_settings=ls_settings)
    data_load = data.load_from_file()
    if data_load is None:
        data.shift_positions()
        settings(**nx_settings)
        jobs = data.get_job_list()
        run_project(jobs)
        data.load_results()
        data.write_to_file()
    else:
        data = data_load
    #end if
    data_ls.append(data)
#end for

if __name__=='__main__':
    surrogate_diagnostics(data_ls)
#end if
