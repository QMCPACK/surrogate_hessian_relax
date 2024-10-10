#!/usr/bin/env python


from surrogate_classes import match_to_tol

from assets import pos_H2O, get_structure_H2O, get_hessian_H2O
from assets import get_surrogate_H2O
from assets import Gs_N200_M7
from assets import job_H2O_pes, analyze_H2O_pes

__author__ = "Juha Tiihonen"
__email__ = "tiihonen@iki.fi"
__license__ = "BSD-3-Clause"


# test LineSearchIteration class
def test_linesearchiteration_class():
    from surrogate_classes import LineSearchIteration
    from shutil import rmtree
    s = get_structure_H2O()
    s.shift_params([0.2, -0.2])
    # params_ref = s.forward(pos_H2O)
    h = get_hessian_H2O()
    # must make global for pickling

    # test deterministic line-search iteration
    test_dir = 'tmp/test_pls_h2O/'
    rmtree(test_dir, ignore_errors=True)
    lsi = LineSearchIteration(
        path=test_dir,
        hessian=h,
        structure=s,
        mode='nexus',
        pes_func=job_H2O_pes,
        load_func=analyze_H2O_pes,
        windows=[0.05, 1.0],
        load=False)
    job_data = lsi.generate_jobs()
    lsi.load_results(load_args={'job_data': job_data})
    lsi.propagate(write=True)
    assert match_to_tol(
        lsi.pls_list[-1].structure.params, [0.89725537, 104.12804938])
    # second iteration
    job_data = lsi.generate_jobs()
    lsi.load_results(load_args={'job_data': job_data})
    lsi.propagate(write=True)
    assert match_to_tol(
        lsi.pls_list[-1].structure.params, [0.93244294, 104.1720672])
    # third iteration
    job_data = lsi.generate_jobs()
    lsi.load_results(load_args={'job_data': job_data})
    lsi.propagate(write=False)
    assert match_to_tol(
        lsi.pls_list[-1].structure.params, [0.93703957, 104.20617541])
    # start over and load until second iteration
    lsi = LineSearchIteration(path=test_dir, load=True)
    assert len(lsi.pls_list) == 2
    lsi.propagate(write=False)
    lsi.generate_jobs()
    lsi.load_results(load_args={'job_data': job_data})
    lsi.propagate(write=False)
    assert match_to_tol(
        lsi.pls_list[-1].structure.params, [0.93703957, 104.20617541])
    rmtree(test_dir)

    # test starting from surrogate
    test_dir = 'tmp/test_pls_srg_H2O/'
    rmtree(test_dir, ignore_errors=True)
    srg = get_surrogate_H2O()
    srg.optimize(windows=[0.1, 0.05], noises=[0.005, 0.005],
                 M=5, Gs=Gs_N200_M7.reshape(2, -1, 5))
    lsi = LineSearchIteration(
        path=test_dir,
        surrogate=srg,
        pes_func=job_H2O_pes,
        load_func=analyze_H2O_pes,
        mode='nexus')
    job_data = lsi.generate_jobs()
    lsi.load_results(load_args={'job_data': job_data})
    lsi.propagate(write=True)
    grid0_ref = [-0.432306, -0.216153, 0., 0.216153, 0.432306]
    grid1_ref = [-0.482330, -0.241165, 0., 0.241165, 0.482330]
    assert match_to_tol(lsi.pls(0).ls(0).grid, grid0_ref, tol=1e-5)
    assert match_to_tol(lsi.pls(0).ls(1).grid, grid1_ref, tol=1e-5)
    assert match_to_tol(lsi.pls().ls(0).grid, grid0_ref, tol=1e-5)
    assert match_to_tol(lsi.pls().ls(1).grid, grid1_ref, tol=1e-5)
    rmtree(test_dir)
    rmtree('tmp/')
# end def
