#! /usr/bin/env python3

from nexus import settings,run_project,obj
from matplotlib import pyplot as plt
from surrogate_error_scan import IterationData,error_scan_diagnostics,load_W_max,scan_error_data,load_of_epsilon,optimize_window_sigma

from parameters import pos_to_params,params_to_pos,nx_settings,get_srg_job,steps_times_error2

from run_phonon import hessian_delta,Lambda
from run_phonon import pos_relax as pos_init

path      = 'scan_error/'
pfn       = 3
pts       = 7
epsilon   = 0.01
fraction  = 0.025
windows   = Lambda
show_plot = False

ls_settings = obj(
    get_jobs      = get_srg_job,
    Finv          = pos_to_params,
    F             = params_to_pos,
    windows       = windows,
    fraction      = fraction,
    pts           = 15,
    path          = path,
    type          = 'scf',
    load_postfix  = '/scf.in',
    colors        = ['r','b','g','c','m','k'],
    targets       = pos_to_params(pos_init),
    )
scan_settings = obj(
    pts       = pts,
    pfn       = pfn,
    generate  = 1000,
    W_num     = 11,
    sigma_num = 11,
    sigma_max = 0.03,
    relative  = False,
    )

# first iteration, try to load data
data = IterationData(n=0, pos=pos_init, hessian=hessian_delta, **ls_settings)
data_load = data.load_from_file()
if data_load is None:
    data.shift_positions()
    settings(**nx_settings)
    jobs = data.get_job_list()
    run_project(jobs)
    data.load_results()
    scan_error_data(data,**scan_settings)
    load_of_epsilon(data,show_plot=show_plot)
    optimize_window_sigma(data,epsilon=epsilon,show_plot=show_plot)
    data.write_to_file()
else:
    data = data_load
#end if

if __name__=='__main__':
    error_scan_diagnostics(data,steps_times_error2)
    plt.show()
#end if
