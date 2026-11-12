import os
import numpy as np
from itertools import product
from datetime import datetime

from paths import plot_path
import plots_and_metrics as pm

#######################################################################################################
############################################## EDIT HERE ##############################################
#######################################################################################################

######################### TRAINING - TESTING PARAMETERS ##################################
TEST_MODE           = 'val'
DATASET             = 'mayo_clinic_512'
TEST_DATASET_RATIO  = 0.05
PROBLEM             = 'lstv'

# TRANING PARAMETERS
DATE                = '2026-11-12'
# DATE                = datetime.today().strftime('%Y-%m-%d')
EPOCHS_TO_SAVE      = [ 20 ]
MINITER             = [ 1 ]
MAXITER             = [ 10 ]
TRAIN_DATASET_RATIO = [ 0.05 ]

# FISTA PARAMETER
GAMMA_FISTA_UNTRAINED  = [ 1.1, 1.5, 2.0, 2.5 ]
ALPHA_FISTA_UNTRAINED  = [ 0.0, 0.25, 0.5, 0.75, 0.9 ]
T_VARIANT_UNTRAINED    = [ 1 ]
PARAMS_FISTA_UNTRAINED = list( product( GAMMA_FISTA_UNTRAINED, ALPHA_FISTA_UNTRAINED, T_VARIANT_UNTRAINED ) )

# FISTA-DEV PARAMETERS
GAMMA     = [ ]
ALPHA     = [ ]
T_VARIANT = [ ]
PARAMS    = list( product( GAMMA, ALPHA, T_VARIANT ) )
ABLATION  = [ 'd', 'e', 'v', 'dev', 'theta', 'all' ]
FISTA_LD_LIST = list( product( PARAMS, TRAIN_DATASET_RATIO, EPOCHS_TO_SAVE, list( zip( MINITER, MAXITER ) ), ABLATION ) )

# DEEPOPT PARAMETER
ALPHA_DEEPOPT = [ ]
DEEPOPT_LIST  = list( product( ALPHA_DEEPOPT, TRAIN_DATASET_RATIO, EPOCHS_TO_SAVE, list( zip( MINITER, MAXITER ) ) ) )

#######################################################################################################
############################################## EDIT HERE ##############################################
#######################################################################################################

######################### DATA PATH ##################################
DATA_PATH = plot_path( TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE )
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

######################### PLOTTING MEDIAN (F-F*)/F* ##################################
#### LOADING F DATA
F_dicts = pm.load_data( 'Fhist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                        PARAMS_FISTA_UNTRAINED, FISTA_LD_LIST, DEEPOPT_LIST )
#### COMPUTING QUANTILES OF (F-F*)/F*
pm.compute_quantiles_rel_opt_gap( F_dicts, DATASET, TEST_MODE, PROBLEM, TEST_DATASET_RATIO )
#### STACKING O(1/k**2)
iters = F_dicts[0].get('data').shape[1]
F_dicts = [
    dict(
        center_line = 6000/(1 + np.arange(iters))**2,
        lower_line = 6000/(1 + np.arange(iters))**2,
        upper_line = 6000/(1 + np.arange(iters))**2,
        label = '$O(1/k^2)$',
        ls = ':',
        c = 'k'
    )
] + F_dicts
#### STACKING THE X-COORDINATE
F_dicts = [
    dict(
        center_line = 1 + np.arange(iters),
        lower_line = 1 + np.arange(iters),
        upper_line = 1 + np.arange(iters),
        label = '$k+1$',
        ls = None,
        c = None
    )
] + F_dicts
#### PLOTTING
optgap_fig = pm.plot_data( F_dicts,
                           ylabel = r'$\frac{F_b(\Tilde{x}_{b,k})-F_b^*}{F_b^*}$',
                           fig_path = DATA_PATH, fig_name = 'optgap' )
#### ADDING MARKS AT THE X-AXIS WHERE THE PROXIMAL OPERATOR SOLVER STARTS TO FAIL
flag_dicts = pm.load_data( 'flag_hist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                           PARAMS_FISTA_UNTRAINED, FISTA_LD_LIST, DEEPOPT_LIST )
pm.add_marks( flag_dicts, optgap_fig, fig_path = DATA_PATH, fig_name = 'optgap_w_marks' )

######################### PLOTTING MEDIAN (F-F*)/F* -- only trained iterations ##################################
#### COPYING (F-F*)/F* DICTS AND UPDATING DATA
F_dicts_zoom = [ d.copy() for d in F_dicts ]
for i,d in enumerate(F_dicts_zoom):
    d.update(
        {
            'center_line':F_dicts_zoom[i].get('center_line')[:max(MAXITER)+1],
            'upper_line':F_dicts_zoom[i].get('upper_line')[:max(MAXITER)+1],
            'lower_line':F_dicts_zoom[i].get('lower_line')[:max(MAXITER)+1]
        } 
    )
#### PLOTTING
pm.plot_data( F_dicts_zoom,
              ylabel = r'$\frac{F_b(\Tilde{x}_{b,k})-F_b^*}{F_b^*}$',
              fig_path = DATA_PATH, fig_name = 'optgapzoom' )

######################### TRAINED ALGORITHMS' MEDIAN VALIDATION LOSS AND PENALTY ##################################
if len(FISTA_LD_LIST) > 0 or len(DEEPOPT_LIST) > 0:
    val_loss, penalty, metrics_label_list = pm.compute_metrics( TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM,
                                                                DATE, FISTA_LD_LIST, DEEPOPT_LIST )
    pm.print_metrics_analysis( val_loss, penalty, metrics_label_list, table_path = DATA_PATH, table_name = 'metrics_table' )

######################### PLOTTING EXTRA INFORMATION ##################################
if not os.path.exists(os.path.join( DATA_PATH, 'extra' )):
    os.makedirs(os.path.join( DATA_PATH, 'extra' ))

######################### PLOTTING MEDIAN STEPSIZE ##################################
#### LOADING STEPSIZE DATA
stepsize_dicts = pm.load_data( 'stepsize_hist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                                PARAMS_FISTA_UNTRAINED, FISTA_LD_LIST, DEEPOPT_LIST )
if len(stepsize_dicts) > 0:
    #### COMPUTING QUANTILES OF STEPSIZE (currently all stepsizes in the batch are the same)
    pm.compute_quantiles( stepsize_dicts )
    #### STACKING THE X-COORDINATE
    iters = stepsize_dicts[0].get('data').shape[1]
    stepsize_dicts = [
        dict(
            center_line = 1 + np.arange(iters),
            lower_line = 1 + np.arange(iters),
            upper_line = 1 + np.arange(iters),
            label = '$k+1$',
            ls = None,
            c = None
        )
    ] + stepsize_dicts
    #### PLOTTING
    pm.plot_data( stepsize_dicts,
                ylabel = r'$\rho_k$',
                fig_path = os.path.join( DATA_PATH, 'extra' ), fig_name = 'stepsize' )

######################### PLOTTING MEDIAN EPS ##################################
#### LOADING EPS DATA
eps_dicts = pm.load_data( 'eps_hist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                        PARAMS_FISTA_UNTRAINED, FISTA_LD_LIST, [] )
if len(eps_dicts) > 0:
    #### COMPUTING MEDIAN EPS
    pm.compute_quantiles( eps_dicts )
    #### STACKING O(1/k**3)
    iters = eps_dicts[0].get('data').shape[1]
    eps_dicts = [
        dict(
            center_line = 1e+3/(1 + np.arange(iters))**2,
            lower_line = 1e+3/(1 + np.arange(iters))**2,
            upper_line = 1e+3/(1 + np.arange(iters))**2,
            label = '$O(1/k^3)$',
            ls = ':',
            c = 'k'
        )
    ] + eps_dicts
    #### STACKING THE X-COORDINATE
    eps_dicts = [
        dict(
            center_line = 1 + np.arange(iters),
            lower_line = 1 + np.arange(iters),
            upper_line = 1 + np.arange(iters),
            label = '$k+1$',
            ls = None,
            c = None
        )
    ] + eps_dicts
    #### PLOTTING
    pm.plot_data( eps_dicts,
                ylabel = r'$\epsilon_k$',
                fig_path = os.path.join( DATA_PATH, 'extra' ), fig_name = 'eps' )

# ######################### DEVIATIONS' INFO ##################################
# d_norm_dicts = []
# d_ub_norm_dicts = []
# #### LOADING DATA -- FISTA-DEV
# for filename in [ 'd_hist', 'e_hist', 'v_hist' ]:
#     d_norm_dicts += pm.load_data( filename, TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE, [], FISTA_LD_LIST, [] )
# for filename in [ 'd_ub_hist', 'e_ub_hist', 'v_ub_hist' ]:
#     d_ub_norm_dicts += pm.load_data( filename, TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE, [], FISTA_LD_LIST, [] )
# #### LOADING DATA -- DEEPOPT
# for filename in [ 'dx1hist', 'dx2hist' ]:
#     d_norm_dicts += pm.load_data( filename, TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE, [], [], DEEPOPT_LIST )
# for filename in [ 'dx1ubhist', 'dx2ubhist' ]:
#     d_ub_norm_dicts += pm.load_data( filename, TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE, [], [], DEEPOPT_LIST)


# ######################### PLOTTING DEVIATION NORMS ##################################
# if len(d_norm_dicts) > 0:
#     #### COMPUTING MEDIAN VALUE
#     d_norms = pm.compute_mean( d_norm_dicts )
#     #### STACKING O(1/k)
#     iters = d_norms[0].get('data').shape[0]
#     d_norms = [ dict( data = 1e+3/(1 + np.arange(iters)), label = '$O(1/k)$', ls = ':', c = 'k' ) ] + d_norms
#     #### STACKING O(1/k**1.5)
#     d_norms = [ dict( data = 3e+2/(1 + np.arange(iters)), label = r'$O(1/k^{1.5})$', ls = '-.', c = 'k' ) ] + d_norms
#     #### STACKING THE X-COORDINATE
#     d_norms = [ dict( data = 1 + np.arange(iters), label = '$k$', ls = None, c = None ) ] + d_norms
#     #### PLOTTING
#     pm.plot_data( d_norms,
#                 ylabel = r'$\mean_b\left(\|\text{deviation}_{b,k}\|_2\right)$',
#                 fig_path = os.path.join( DATA_PATH, 'extra' ), fig_name = 'deviations_norm' )

# ######################### PLOTTING UPPER BOUNDS TO DEVIATION NORMS ##################################
# if len(d_ub_norm_dicts) > 0:
#     #### COMPUTING MEDIAN VALUE
#     d_ub_norms = pm.compute_mean( d_ub_norm_dicts )
#     #### STACKING O(1/k)
#     iters = d_ub_norms[0].get('data').shape[0]
#     d_ub_norms = [ dict( data = 1e+3/(1 + np.arange(iters)), label = '$O(1/k)$', ls = ':', c = 'k' ) ] + d_ub_norms
#     #### STACKING O(1/k**1.5)
#     d_ub_norms = [ dict( data = 3e+2/(1 + np.arange(iters)), label = r'$O(1/k^{1.5})$', ls = '-.', c = 'k' ) ] + d_ub_norms
#     #### STACKING THE X-COORDINATE
#     d_ub_norms = [ dict( data = 1 + np.arange(iters), label = '$k$', ls = None, c = None ) ] + d_ub_norms
#     #### PLOTTING
#     pm.plot_data( d_ub_norms,
#                 ylabel = r'$\mean_b\left(\|\text{deviation}_{b,k}\|_2\text{ upper bound}\right)$',
#                 fig_path = os.path.join( DATA_PATH, 'extra' ), fig_name = 'deviations_norm_upper_bound' )

# ######################### PLOTTING DEVIATION NORMS REL TO UPPER BOUNDS ##################################
# if len(d_ub_norm_dicts) > 0 and len(d_norm_dicts) > 0:
#     #### COMPUTING THE RATIO DEVIATION NORM / UPPER BOUND
#     d_ratio_norm_dicts = [ d.copy() for d in d_norm_dicts ]
#     for i,d in enumerate(d_ratio_norm_dicts):
#         d.update( { 'data': d_norm_dicts[i].get('data') / d_ub_norm_dicts[i].get('data') } )
#     #### COMPUTING MEDIAN VALUE
#     d_ratio_norms = pm.compute_mean( d_ratio_norm_dicts )
#     #### STACKING THE X-COORDINATE
#     iters = d_ub_norms[0].get('data').shape[0]
#     d_ratio_norms = [ dict( data = 1 + np.arange(iters), label = '$k$', ls = None, c = None ) ] + d_ratio_norms
#     #### PLOTTING
#     pm.plot_data( d_ratio_norms,
#                 ylabel = r'$\mean_b\left( \frac{\|\text{deviation}_{b,k}\|_2}{\|\text{deviation}_{b,k}\|_2\text{ upper bound}} \right)$',
#                 fig_path = os.path.join( DATA_PATH, 'extra' ), fig_name = 'deviations_ratio', yscale = 'linear' )

######################### PLOTTING MEDIAN INNER ITERS ##################################
#### LOADING INNER ITERS DATA
inner_iter_dicts = pm.load_data( 'inner_iter_hist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE, 
                                 PARAMS_FISTA_UNTRAINED, FISTA_LD_LIST, DEEPOPT_LIST )
if len(inner_iter_dicts) > 0:
    #### COMPUTING QUANTILES
    pm.compute_quantiles( inner_iter_dicts )
    #### STACKING O(k**2)
    iters = inner_iter_dicts[0].get('data').shape[1]
    inner_iter_dicts = [
        dict(
            center_line = np.concatenate( [ (1 + np.arange(10))**2, [np.nan]*(iters-10) ] ),
            lower_line = np.concatenate( [ (1 + np.arange(10))**2, [np.nan]*(iters-10) ] ),
            upper_line = np.concatenate( [ (1 + np.arange(10))**2, [np.nan]*(iters-10) ] ),
            label = '$O(k^2)$',
            ls = ':',
            c = 'k'
        )
    ] + inner_iter_dicts
    #### STACKING THE X-COORDINATE
    inner_iter_dicts = [
        dict(
            center_line = 1 + np.arange(iters),
            lower_line = 1 + np.arange(iters),
            upper_line = 1 + np.arange(iters),
            label = '$k$',
            ls = None,
            c = None
        )
    ] + inner_iter_dicts
    #### PLOTTING
    pm.plot_data( inner_iter_dicts,
                ylabel = r'$\text{inner\_iters}_{k}$',
                fig_path = os.path.join( DATA_PATH, 'extra' ), fig_name = 'inner_iters' )
    
######################### PLOTTING MEDIAN CUMULATIVE INNER ITERS ##################################
#### LOADING INNER ITERS DATA
inner_iter_dicts = pm.load_data( 'inner_iter_hist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE, 
                                 PARAMS_FISTA_UNTRAINED, FISTA_LD_LIST, DEEPOPT_LIST )
if len(inner_iter_dicts) > 0:
    #### COMPUTING THE COMULATIVE INNER ITERS
    cum_inner_iter_dicts = [ d.copy() for d in inner_iter_dicts ]
    for i,d in enumerate(cum_inner_iter_dicts):
        d.update( { 'data': np.cumsum( inner_iter_dicts[i].get('data'), axis = 1 ) } )
    #### COMPUTING MEDIAN VALUE
    pm.compute_quantiles( cum_inner_iter_dicts )
    #### STACKING THE X-COORDINATE
    iters = cum_inner_iter_dicts[0].get('data').shape[1]
    cum_inner_iter_dicts = [
        dict(
            center_line = 1 + np.arange(iters),
            lower_line = 1 + np.arange(iters),
            upper_line = 1 + np.arange(iters),
            label = '$k$',
            ls = None,
            c = None
        )
    ] + cum_inner_iter_dicts
    #### PLOTTING
    pm.plot_data( cum_inner_iter_dicts,
                  ylabel = r'$\text{cum\_inner\_iters}_{k}$',
                  fig_path = os.path.join( DATA_PATH, 'extra' ), fig_name = 'cum_inner_iters' )
    
# ######################### PERFORMANCE PROFILE ##################################
# #### SETTING TOLERANCE FOR (F-F*)/F*
# tol = 1e-1
# #### LOADING F DATA
# F_dicts = pm.load_data( 'Fhist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
#                         PARAMS_FISTA_UNTRAINED, FISTA_LD_LIST, DEEPOPT_LIST )
# if len(F_dicts) > 1:
#     #### COMPUTING F*
#     F_hist = np.stack( [ d.get('data') for d in F_dicts ] )
#     F_opt = np.min( np.min( F_hist , axis = 2 ), axis = 0 )
#     F_opt = F_opt[:,None]
#     #### COMPUTING THE FIRST ITERATION IN WHICH THE TOLERANCE IS MET
#     perf_list = []
#     for i,d in enumerate(F_dicts):
#         mask = ( F_dicts[i].get('data') - F_opt ) / F_opt < tol
#         tol_iter = np.inf * np.ones( (mask.shape[0],) )
#         for j,m in enumerate(mask):
#             tol_iter[j] = np.argmax(m) if np.any(m) else np.inf
#         perf_list.append(tol_iter)
#     #### PERFORMANCE MATRIX
#     perf_mat = np.stack( perf_list )
#     algorithms, problem_instances = perf_mat.shape
#     #### COMPUTING KAPPAS
#     min_perf_mat = np.min( perf_mat, axis = 0 )
#     max_perf_mat = np.max( perf_mat, axis = 0, where=perf_mat!=np.inf, initial=-1.0 )
#     kappas = np.linspace( 1, 1.01 * np.max( max_perf_mat / min_perf_mat ), num = 500 ) # multiply by 1.01 to avoid missing the maxcost in the plot by numerical reasons
#     #### COMPUTING THE FRACTION OF PROBLEM INSTANCES FOR EACH KAPPA
#     fractions_to_plot = np.zeros( (algorithms, kappas.size) )
#     for i in range(algorithms):
#         for l,kappa in enumerate(kappas):
#             fractions_to_plot[i,l] = np.sum( perf_mat[i] <= kappa * min_perf_mat ) / problem_instances
#     #### CREATING DICTIONARIES
#     perf_dicts = [ d.copy() for d in F_dicts ]
#     for i,d in enumerate(perf_dicts):
#         d.update( { 'data': fractions_to_plot[i] } )
#     #### STACKING THE X-COORDINATE (KAPPAS)
#     perf_dicts = [ dict( data = kappas, label = r'$\kappa$', ls = None, c = None ) ] + perf_dicts
#     #### PLOTTING
#     pm.plot_data( perf_dicts,
#                   ylabel = r'$\Gamma(\kappa)$',
#                   fig_path = os.path.join( DATA_PATH, 'extra' ),
#                   fig_name = 'perf_profile',
#                   xscale = 'linear', yscale = 'linear' )

print( 'All data was saved to', DATA_PATH )