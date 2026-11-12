import os
import numpy as np
import matplotlib.pyplot as pp
import matplotlib.colors as mcolors
import pandas as pd

from paths import hist_path
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from data import get_dataset

######################### PLOTTING OPTIONS ##################################
pp.rc( 'text', usetex = True )
pp.rc( 'font', size = 12, family = 'serif' )
pp.rcParams[ 'text.latex.preamble' ] = r"\usepackage{amsmath}\DeclareMathOperator{\mean}{mean}"
pp.rcParams[ 'savefig.bbox' ] = 'tight'
EXTENSION = '.pdf'
COLORS = list(mcolors.TABLEAU_COLORS.keys())

#######################################################################################################################
################################################ AUXILIARY FUNCTIONS ##################################################
#######################################################################################################################

def get_label( algorithm, list_of_parameters ):
    if algorithm == 'ista':
        label = 'ISTA'
    elif algorithm == 'fista':
        gamma, alpha, t_variant = list_of_parameters
        label = f'FISTA: $\\gamma={gamma:.2e}$, $\\alpha={alpha:.2e}$, $T_k$ variant:{t_variant}'
    elif algorithm == 'fista-dev':
        train_dataset_ratio, miniter, maxiter, gamma, alpha, ablation, t_variant, epochs = list_of_parameters
        label = rf'fista-dev: N=[{miniter},{maxiter}], '
        label += f'$\\gamma={gamma:.2e}$, $\\alpha={alpha:.2e}$, $T_k$ variant:{t_variant}, '
        label += f'train data={train_dataset_ratio}, epochs={epochs}, ablation={ablation}'
    elif algorithm == 'deepopt':
        train_dataset_ratio, miniter, maxiter, alpha, epochs = list_of_parameters
        label = f'DeepOpt: N=[{miniter},{maxiter}], $\\alpha$={alpha:.2f}, '
        label += f'train data={train_dataset_ratio}, epochs={epochs}'
    else:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    return label

def load_data( data_name, test_mode, dataset, test_dataset_ratio, problem, date, fista_list, fista_ld_list, deepopt_list ):
    dicts_list = []
    if data_name in ['Fhist', 'flag_hist', 'inner_iter_hist', 'eps_hist', 'stepsize_hist']:
        ##### ISTA RESULTS
        hist_path_ista = hist_path( 'ista', [ test_mode, dataset, test_dataset_ratio, problem ] )
        dicts_list.append(
            dict(
                data  = np.load( os.path.join( hist_path_ista, data_name + '.npy' ) ),
                label = 'ISTA',
                ls    = '-.',
                c     = 'k'
            )
        )
        ##### FISTA RESULTS
        for i,params in enumerate(fista_list):
            gamma, alpha, t_variant = params
            hist_path_fista = hist_path( 'fista', [ test_mode, dataset, test_dataset_ratio, problem, gamma, alpha, t_variant ] )
            dicts_list.append(
                dict(
                    data  = np.load( os.path.join( hist_path_fista, data_name + '.npy' ) ),
                    label = get_label( 'fista', params ),
                    ls    = '-.',
                    c     = COLORS[i%len(COLORS)]
                )
            )
    #### FISTA-DEV RESULTS
    for i,options in enumerate(fista_ld_list):
        params, train_dataset_ratio, epochs, minmaxiter, ablation = options
        gamma, alpha, t_variant = params
        miniter, maxiter = minmaxiter
        hist_path_fista_ld = hist_path( 'fista-dev', [ test_mode, dataset, train_dataset_ratio, test_dataset_ratio, 
                                                      problem, miniter, maxiter, gamma, alpha, date, ablation, t_variant, epochs ] )
        dicts_list.append(
            dict(
                data  = np.load( os.path.join( hist_path_fista_ld, data_name + '.npy') ),
                label = get_label( 'fista-dev', [ train_dataset_ratio, miniter, maxiter, gamma, alpha, ablation, t_variant, epochs ] ),
                ls    = '-',
                c     = COLORS[i%len(COLORS)]
            )
        )
    #### DEEPOPT RESULTS
    for i,options in enumerate(deepopt_list):
        alpha, train_dataset_ratio, epochs, minmaxiter = options
        miniter, maxiter = minmaxiter
        hist_path_deepopt = hist_path( 'deepopt', [ test_mode, dataset, train_dataset_ratio, test_dataset_ratio,
                                                    problem, miniter, maxiter, alpha, date, epochs ] )
        dicts_list.append(
            dict(
                data  = np.load( os.path.join( hist_path_deepopt, data_name + '.npy') ),
                label = get_label( 'deepopt', [ train_dataset_ratio, miniter, maxiter, alpha, epochs ] ),
                ls    = '--',
                c     = COLORS[i%len(COLORS)]
            )
        )
    return dicts_list

def compute_quantiles( dicts_list ):
    for d in dicts_list:
        q25, med, q75 = np.quantile( d.get('data'), [0.25, 0.5, 0.75], axis = 0 )
        d['center_line'] = med
        d['upper_line']  = q75
        d['lower_line']  = q25

def save_data( dicts_list, data_path, data_name ):
    center_line_mat = np.stack( [ d.get('center_line') for d in dicts_list ] ).T
    upper_line_mat = np.stack( [ d.get('upper_line') for d in dicts_list ] ).T
    lower_line_mat = np.stack( [ d.get('lower_line') for d in dicts_list ] ).T
    np.savetxt( os.path.join( data_path, data_name + '_center.txt' ), center_line_mat )
    np.savetxt( os.path.join( data_path, data_name + '_upper.txt' ), upper_line_mat )
    np.savetxt( os.path.join( data_path, data_name + '_lower.txt' ), lower_line_mat )
    with open( os.path.join( data_path, data_name + '_labels.txt' ), 'w' ) as f:
        for d in dicts_list:
            f.write(f"{d.get('label')}\n")

def plot_data( dicts_list, ylabel, fig_path, fig_name, xscale = 'log', yscale = 'log' ):
    x = dicts_list[0]
    for y in dicts_list[1:]:
        pp.plot( x.get('center_line'), y.get('center_line'), label = y.get('label'), ls = y.get('ls'), c = y.get('c') )
        pp.fill_between( x.get('center_line'), y.get('lower_line'), y.get('upper_line'), color = y.get('c'), alpha = 0.3, linewidth = 0 )
    pp.xscale( xscale )
    pp.yscale( yscale )
    pp.xlabel( x.get('label') )
    pp.ylabel( ylabel )
    pp.grid()
    pp.legend( loc = 'center left', bbox_to_anchor = ( 1, 0.5 ) )
    pp.savefig( os.path.join( fig_path, fig_name + EXTENSION ) )
    fig = pp.gcf()
    pp.close()
    save_data( dicts_list, data_path = fig_path, data_name = fig_name )
    return fig

def add_marks( dicts_list, fig, fig_path, fig_name ):
    flag_iter_dicts = [ d.copy() for d in dicts_list ]
    for i,d in enumerate(flag_iter_dicts):
        mask = np.logical_not( dicts_list[i].get('data') )
        fail_iter = np.zeros( (mask.shape[0],) )
        for j,m in enumerate(mask):
            fail_iter[j] = np.argmax(m) if np.any(m) else np.inf
        if np.all( fail_iter == np.inf ):
            d['center_line'] = -1
            d['lower_line'] = -1
            d['upper_line'] = -1
        else:
            q25, med, q75 = np.quantile( fail_iter, [0.25, 0.5, 0.75] )
            d['center_line'] = np.round(med)
            d['lower_line'] = np.round(q25)
            d['upper_line'] = np.round(q75)
    ax = fig.gca()
    for d in flag_iter_dicts:
        ax.axvline( d.get('center_line'), alpha = 0.5, label = f'prox failure ' + d.get('label'), ls = d.get('ls'), c = d.get('c') )
    fig.savefig( os.path.join( fig_path, fig_name + EXTENSION ) )
    save_data( flag_iter_dicts, fig_path, 'prox_fail' )

def compute_quantiles_rel_opt_gap( F_dicts, dataset, test_mode, problem, test_dataset_ratio ):
    #### TRAINING DATASET
    ds = get_dataset( dataset, test_mode, problem, batch_size = 1, F_ref = None, overall_ratio = test_dataset_ratio )
    F_train_hist_list = []
    for (_, F_train) in ds:
        F_train_hist_list.append( F_train.numpy()[0] )
    # Reference functional values in training
    F_train_hist = np.stack( F_train_hist_list )
    F_opt_train = np.min( F_train_hist, axis = 1 )
    #### COMPUTING F* FROM DATA
    F_hist = np.stack( [ d.get('data') for d in F_dicts ] )
    F_opt_data = np.min( np.min( F_hist , axis = 2 ), axis = 0 )
    #### COMPUTING BEST F*
    F_opt = np.minimum( F_opt_data, F_opt_train )[:,None]
    #### COMPUTING (F-F*)/F*
    for d in F_dicts:
        d['data'] -= F_opt
        d['data'] /= F_opt
    #### COMPUTING QUANTILES
    compute_quantiles( F_dicts )

def compute_metrics( test_mode, dataset, test_dataset_ratio, problem, date, fista_ld_list, deepopt_list ):
    val_loss_list = []
    penalty_list = []
    label_list = []
    #### TRAINING DATASET
    ds = get_dataset( dataset, test_mode, problem, batch_size = 1, F_ref = None, overall_ratio = test_dataset_ratio )
    F_train_hist_list = []
    for (_, F_train) in ds:
        F_train_hist_list.append( F_train.numpy()[0] )
    # Reference functional values in training
    F_train_hist = np.stack( F_train_hist_list )
    F_opt_train = np.min( F_train_hist, axis = 1 )[:,None]
    #### FISTA-DEV RESULTS
    for options in fista_ld_list:
        params, train_dataset_ratio, epochs, minmaxiter, ablation = options
        gamma, alpha = params
        miniter, maxiter = minmaxiter
        hist_path_fista_ld = hist_path( 'fista-dev', [ test_mode, dataset, train_dataset_ratio, test_dataset_ratio, 
                                                      problem, miniter, maxiter, gamma, alpha, date, ablation, epochs ] )
        hist = np.load( os.path.join( hist_path_fista_ld, 'Fhist.npy') )
        val_loss_list.append( np.mean( ( hist[:,miniter:maxiter+10+1] - F_opt_train ) / ( F_train_hist[:,miniter:maxiter+10+1] - F_opt_train ) ) )
        penalty_max_iters = min( F_train_hist.shape[1], hist.shape[1] )
        penalty_list.append( np.mean( np.maximum( (hist[:,maxiter+10+1:penalty_max_iters] - F_train_hist[:,maxiter+10+1:penalty_max_iters])/F_train_hist[:,maxiter+10+1:penalty_max_iters], 0.0 ) ) )
        label_list.append( get_label( 'fista-dev', [ train_dataset_ratio, miniter, maxiter, gamma, alpha, ablation, epochs ] ) )
    #### DEEPOPT RESULTS
    for options in deepopt_list:
        alpha, train_dataset_ratio, epochs, minmaxiter = options
        miniter, maxiter = minmaxiter
        hist_path_deepopt = hist_path( 'deepopt', [ test_mode, dataset, train_dataset_ratio, test_dataset_ratio,
                                                    problem, miniter, maxiter, alpha, date, epochs ] )
        hist = np.load( os.path.join( hist_path_deepopt, 'Fhist.npy') )
        val_loss_list.append( np.mean( ( hist[:,miniter:maxiter+10+1] - F_opt_train ) / ( F_train_hist[:,miniter:maxiter+10+1] - F_opt_train ) ) )
        penalty_max_iters = min( F_train_hist.shape[1], hist.shape[1] )
        penalty_list.append( np.mean( np.maximum( (hist[:,maxiter+10+1:penalty_max_iters] - F_train_hist[:,maxiter+10+1:penalty_max_iters])/F_train_hist[:,maxiter+10+1:penalty_max_iters], 0.0 ) ) )
        label_list.append( get_label( 'deepopt', [ train_dataset_ratio, miniter, maxiter, alpha, epochs ] ) )
    #### GENERAL ANALYSIS
    val_loss = np.stack( val_loss_list )
    penalty = np.stack( penalty_list )
    return val_loss, penalty, label_list

def print_metrics_analysis( val_loss, penalty, label_list, table_path = None, table_name = None ):
    print('Metrics analysis:')
    print( f'Best validation loss: {np.min(val_loss):.2e}' )
    print( f'Best algorithm w.r.t. validation loss: {label_list[np.argmin(val_loss)]}' )
    print( f'Best penalty: {np.min(penalty):.2e}' )
    print( f'Best parameters w.r.t. penalty: {label_list[np.argmin(penalty)]}' )
    if table_path is not None and table_name is not None:
        table = pd.DataFrame( np.stack( [val_loss, penalty] ).T, index = label_list, columns = [ 'Validation loss', 'Penalty' ] )
        table.index.name = 'Algorithm'
        table.columns.name = 'Metric'
        print( 'Table:' )
        print(table.to_string( float_format = "%.2e", col_space = 12 ), '\n')
        with open( os.path.join( table_path, table_name + '.txt' ), 'w' ) as f:
            f.write(table.to_string( float_format = "%.2e" ) )
            # f.write(table.to_latex( float_format="%.2e" ) )