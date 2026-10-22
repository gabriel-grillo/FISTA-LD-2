import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from datetime import datetime
import numpy as np
from itertools import product
import pandas as pd

import plots_and_metrics as pm

######################### DATE OF TRAINING AND TESTING ##################################
# DATE = datetime.today().strftime('%Y-%m-%d')
DATE = '0'

#######################################################################################################################
#################################################### EXPERIMENT 1 #####################################################
#######################################################################################################################

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 1 (Start) ----------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

######################### TRAINING - TESTING PARAMETERS ##################################
TEST_MODE           = 'val'
DATASET             = 'mayo_clinic_512'
TEST_DATASET_RATIO  = 1.0
PROBLEM             = 'lasso'

# TRANING PARAMETERS
EPOCHS_TO_SAVE      = [ 20 ]
MINITER             = [  1 ]
MAXITER             = [ 20 ]
TRAIN_DATASET_RATIO = [ 0.05 ]

# FISTA PARAMETER
TAU_FISTA_UNTRAINED = [ 1.0 ]

# FISTA-LD PARAMETERS
ALPHA         = np.logspace(-3.5, -1.5, 5)
GAMMA         = [ 0.01, 0.025, 0.05, 0.075, 0.1 ]
CONST_TAU     = [ -1.0, -1.0, -1.0, -1.0, -1.0 ]
PARAMS        = list( product( list( zip( GAMMA, CONST_TAU ) ), ALPHA ) )
FISTA_LD_LIST = list( product( PARAMS, TRAIN_DATASET_RATIO, EPOCHS_TO_SAVE, list( zip( MINITER, MAXITER ) ) ) )

# DEEPOPT PARAMETER
ALPHA_DEEPOPT = [ 0.1, 0.25, 0.5, 0.75, 0.9 ]
DEEPOPT_LIST  = list( product( ALPHA_DEEPOPT, TRAIN_DATASET_RATIO, EPOCHS_TO_SAVE, list( zip( MINITER, MAXITER ) ) ) )

######################### DATA PATH ##################################
DATA_PATH = os.path.join(os.getcwd(), 'paper_results_data', 'exp1' )
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

######################### PLOTTING MEAN (F-F*)/F* ##################################
#### LOADING F DATA
F_dicts = pm.load_data( 'Fhist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                        TAU_FISTA_UNTRAINED, FISTA_LD_LIST, DEEPOPT_LIST )
#### COMPUTING MEAN (F-F*)/F*
mean_rel_opt_gap_dicts = pm.compute_mean_rel_opt_gap( F_dicts )
#### STACKING O(1/k**2)
iters = mean_rel_opt_gap_dicts[0].get('data').shape[0]
mean_rel_opt_gap_dicts = [ dict( data = 6000/(1 + np.arange(iters))**2, label = '$O(1/k^2)$', ls = ':', c = 'k' ) ] + mean_rel_opt_gap_dicts
#### STACKING THE X-COORDINATE
mean_rel_opt_gap_dicts = [ dict( data = 1 + np.arange(iters), label = '$k+1$', ls = None, c = None ) ] + mean_rel_opt_gap_dicts
#### SAVING RESULTS
pm.save_data( mean_rel_opt_gap_dicts, data_path = DATA_PATH, data_name = 'optgap' )
#### PLOTTING
for j in range(6):
    pm.plot_data( mean_rel_opt_gap_dicts[:4] + mean_rel_opt_gap_dicts[4+j*5:4+(j+1)*5],
                ylabel = r'$\mean_b\left(\frac{F_b(x_{b,k})-F_b^*}{F_b^*}\right)$',
                fig_path = DATA_PATH, fig_name = f'optgap_{j}' )


######################### COMPUTING TRAINED ALGORITHMS' MEAN VALIDATION LOSS AND PENALTY ##################################
val_loss, penalty, metrics_label_list = pm.compute_metrics( TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM,
                                                            DATE, FISTA_LD_LIST, DEEPOPT_LIST )

#### GENERAL ANALYSIS
print('General:')
pm.print_metrics_analysis( val_loss, penalty, metrics_label_list, table_path = DATA_PATH, table_name = 'metrics_table_all' )

#### FISTA-LD ANALYSIS
# Comparing algorithms
print('FISTA-LD:')
pm.print_metrics_analysis( val_loss[:25], penalty[:25], metrics_label_list[:25] )
# Creating and printing tables
val_loss_matrix_fista_ld = np.reshape( val_loss[:25], (5,5), order = 'F' )
penaly_matrix_fista_ld = np.reshape( penalty[:25], (5,5), order = 'F' )
table_val_loss_fista_ld = pd.DataFrame( val_loss_matrix_fista_ld, index = [ '10^{' +  rf'{np.log10(a):.1f}' + '}' for a in ALPHA], columns = [ f'{a:.3f}' for a in GAMMA] )
table_val_loss_fista_ld.index.name = 'alpha'
table_val_loss_fista_ld.columns.name = 'gamma'
print( '\nValidation loss table for FISTA-LD:' )
print(table_val_loss_fista_ld.to_string( float_format = "%.2e", col_space = 12 ), '\n')
with open( os.path.join( DATA_PATH, 'metrics_tables_fista_ld.txt' ), 'w' ) as f:
    f.write( 'Validation loss table:\n' )
    f.write(table_val_loss_fista_ld.to_string( float_format = "%.2e" ) )
    # f.write(table_val_loss_fista_ld.to_latex( float_format="%.2e" ) )
table_penalty_fista_ld = pd.DataFrame( penaly_matrix_fista_ld, index = [ '10^{' +  rf'{np.log10(a):.1f}' + '}' for a in ALPHA], columns = [ f'{a:.3f}' for a in GAMMA] )
table_penalty_fista_ld.index.name = 'alpha'
table_penalty_fista_ld.columns.name = 'gamma'
print( 'Penalty table for FISTA-LD:' )
print(table_penalty_fista_ld.to_string( float_format = "%.2e", col_space = 12 ), '\n')
with open( os.path.join( DATA_PATH, 'metrics_tables_fista_ld.txt' ), 'a' ) as f:
    f.write( '\n\n\nPenalty table:\n' )
    f.write(table_penalty_fista_ld.to_string( float_format = "%.2e" ) )
    # f.write(table_penalty_fista_ld.to_latex( float_format="%.2e" ) )

#### DEEPOPT ANALYSIS
# Comparing algorithms
print('DeepOpt:')
pm.print_metrics_analysis( val_loss[25:], penalty[25:], metrics_label_list[25:], table_path = DATA_PATH, table_name = 'metrics_table_deepopt' )

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 1 (End) ------------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#######################################################################################################################
#################################################### EXPERIMENT 2 #####################################################
#######################################################################################################################

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 2 (Start) ----------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

######################### TRAINING - TESTING PARAMETERS ##################################
TEST_MODE           = 'val'
DATASET             = 'mayo_clinic_512'
TEST_DATASET_RATIO  = 1.0
PROBLEM             = 'lasso'

# TRANING PARAMETERS
EPOCHS_TO_SAVE      = [ 20, 40, 60, 80, 100 ]
MINITER             = [  1 ]
MAXITER             = [ 20 ]
TRAIN_DATASET_RATIO = [ 1.0 ]

# FISTA PARAMETER
TAU_FISTA_UNTRAINED = [ 1.0 ]

# FISTA-LD PARAMETERS
ALPHA         = [ 10**(-3.0) ]
GAMMA         = [ 0.025 ]
CONST_TAU     = [ -1.0 ]
PARAMS        = list( product( list( zip( GAMMA, CONST_TAU ) ), ALPHA ) )
FISTA_LD_LIST = list( product( PARAMS, TRAIN_DATASET_RATIO, EPOCHS_TO_SAVE, list( zip( MINITER, MAXITER ) ) ) )

# DEEPOPT
DEEPOPT_LIST = []

######################### DATA PATH ##################################
DATA_PATH = os.path.join(os.getcwd(), 'paper_results_data', 'exp2' )
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

######################### PLOTTING MEAN (F-F*)/F* ##################################
#### LOADING F DATA
F_dicts = pm.load_data( 'Fhist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                        TAU_FISTA_UNTRAINED, FISTA_LD_LIST, DEEPOPT_LIST )
#### COMPUTING MEAN (F-F*)/F*
mean_rel_opt_gap_dicts = pm.compute_mean_rel_opt_gap( F_dicts )
#### STACKING O(1/k**2)
iters = mean_rel_opt_gap_dicts[0].get('data').shape[0]
mean_rel_opt_gap_dicts = [ dict( data = 6000/(1 + np.arange(iters))**2, label = '$O(1/k^2)$', ls = ':', c = 'k' ) ] + mean_rel_opt_gap_dicts
#### STACKING THE X-COORDINATE
mean_rel_opt_gap_dicts = [ dict( data = 1 + np.arange(iters), label = '$k+1$', ls = None, c = None ) ] + mean_rel_opt_gap_dicts
#### PLOTTING
pm.plot_data( mean_rel_opt_gap_dicts,
              ylabel = r'$\mean_b\left(\frac{F_b(x_{b,k})-F_b^*}{F_b^*}\right)$',
              fig_path = DATA_PATH, fig_name = 'optgap' )

######################### TRAINED ALGORITHMS' MEAN VALIDATION LOSS AND PENALTY ##################################
val_loss, penalty, metrics_label_list = pm.compute_metrics( TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM,
                                                            DATE, FISTA_LD_LIST, DEEPOPT_LIST )
pm.print_metrics_analysis( val_loss, penalty, metrics_label_list, table_path = DATA_PATH, table_name = 'metrics_table' )

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 2 (End) ------------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#######################################################################################################################
#################################################### EXPERIMENT 3 #####################################################
#######################################################################################################################

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 3 (Start) ----------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')
######################### TRAINING - TESTING PARAMETERS ##################################
TEST_MODE           = 'test'
DATASET             = 'mayo_clinic_512'
TEST_DATASET_RATIO  = 1.0
PROBLEM             = 'lasso'

# TRANING PARAMETERS
EPOCHS_TO_SAVE      = [ 20 ]
MINITER             = [  1 ]
MAXITER             = [ 20 ]
TRAIN_DATASET_RATIO = [ 1.0 ]

# FISTA PARAMETER
TAU_FISTA_UNTRAINED = [ 1.0 ]

# FISTA-LD PARAMETERS
ALPHA         = [ 10**(-3.0) ]
GAMMA         = [ 0.025 ]
CONST_TAU     = [ -1.0 ]
PARAMS        = list( product( list( zip( GAMMA, CONST_TAU ) ), ALPHA ) )
FISTA_LD_LIST = list( product( PARAMS, TRAIN_DATASET_RATIO, EPOCHS_TO_SAVE, list( zip( MINITER, MAXITER ) ) ) )

# DEEPOPT PARAMETER
ALPHA_DEEPOPT = [ 0.5 ]
DEEPOPT_LIST  = list( product( ALPHA_DEEPOPT, TRAIN_DATASET_RATIO, EPOCHS_TO_SAVE, list( zip( MINITER, MAXITER ) ) ) )

######################### DATA PATH ##################################
DATA_PATH = os.path.join(os.getcwd(), 'paper_results_data', 'exp3' )
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

######################### PLOTTING MEAN (F-F*)/F* ##################################
#### LOADING F DATA
F_dicts = pm.load_data( 'Fhist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                        TAU_FISTA_UNTRAINED, FISTA_LD_LIST, DEEPOPT_LIST )
#### COMPUTING MEAN (F-F*)/F*
mean_rel_opt_gap_dicts = pm.compute_mean_rel_opt_gap( F_dicts )
#### STACKING O(1/k**2)
iters = mean_rel_opt_gap_dicts[0].get('data').shape[0]
mean_rel_opt_gap_dicts = [ dict( data = 6000/(1 + np.arange(iters))**2, label = '$O(1/k^2)$', ls = ':', c = 'k' ) ] + mean_rel_opt_gap_dicts
#### STACKING THE X-COORDINATE
mean_rel_opt_gap_dicts = [ dict( data = 1 + np.arange(iters), label = '$k+1$', ls = None, c = None ) ] + mean_rel_opt_gap_dicts
#### PLOTTING
pm.plot_data( mean_rel_opt_gap_dicts,
              ylabel = r'$\mean_b\left(\frac{F_b(x_{b,k})-F_b^*}{F_b^*}\right)$',
              fig_path = DATA_PATH, fig_name = 'optgap' )

######################### TRAINED ALGORITHMS' MEAN VALIDATION LOSS AND PENALTY ##################################
val_loss, penalty, metrics_label_list = pm.compute_metrics( TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM,
                                                            DATE, FISTA_LD_LIST, DEEPOPT_LIST )
pm.print_metrics_analysis( val_loss, penalty, metrics_label_list, table_path = DATA_PATH, table_name = 'metrics_table' )

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 3 (End) ------------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#######################################################################################################################
#################################################### EXPERIMENT 4 #####################################################
#######################################################################################################################

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 4 (Start) ----------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')
######################### TRAINING - TESTING PARAMETERS ##################################
TEST_MODE           = 'val'
DATASET             = 'mayo_clinic_512'
TEST_DATASET_RATIO  = 1.0
PROBLEM             = 'lstv'

# TRANING PARAMETERS
EPOCHS_TO_SAVE      = [ 20 ]
MINITER             = [  1 ]
MAXITER             = [ 20 ]
TRAIN_DATASET_RATIO = [ 0.05 ]

# FISTA PARAMETER
TAU_FISTA_UNTRAINED = [ 0.1, 0.25, 0.5, 0.75, 0.9 ]

# FISTA-LD PARAMETERS
ALPHA         = [ 10**(-3.0) ]
GAMMA         = [ -1.0, -1.0, -1.0, -1.0, -1.0 ]
CONST_TAU     = [ 0.1, 0.25, 0.5, 0.75, 0.9 ]
PARAMS        = list( product( list( zip( GAMMA, CONST_TAU ) ), ALPHA ) )
FISTA_LD_LIST = list( product( PARAMS, TRAIN_DATASET_RATIO, EPOCHS_TO_SAVE, list( zip( MINITER, MAXITER ) ) ) )

# DEEPOPT PARAMETER
ALPHA_DEEPOPT = []
DEEPOPT_LIST  = list( product( ALPHA_DEEPOPT, TRAIN_DATASET_RATIO, EPOCHS_TO_SAVE, list( zip( MINITER, MAXITER ) ) ) )

######################### DATA PATH ##################################
DATA_PATH = os.path.join(os.getcwd(), 'paper_results_data', 'exp4' )
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

######################### PLOTTING MEAN (F-F*)/F* ##################################
#### LOADING F DATA
F_dicts = pm.load_data( 'Fhist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                        TAU_FISTA_UNTRAINED, FISTA_LD_LIST, DEEPOPT_LIST )
#### COMPUTING MEAN (F-F*)/F*
mean_rel_opt_gap_dicts = pm.compute_mean_rel_opt_gap( F_dicts )
#### STACKING O(1/k**2)
iters = mean_rel_opt_gap_dicts[0].get('data').shape[0]
mean_rel_opt_gap_dicts = [ dict( data = 6000/(1 + np.arange(iters))**2, label = '$O(1/k^2)$', ls = ':', c = 'k' ) ] + mean_rel_opt_gap_dicts
#### STACKING THE X-COORDINATE
mean_rel_opt_gap_dicts = [ dict( data = 1 + np.arange(iters), label = '$k+1$', ls = None, c = None ) ] + mean_rel_opt_gap_dicts
#### PLOTTING
optgap_fig_exp4 = pm.plot_data( mean_rel_opt_gap_dicts, 
                                ylabel = r'$\mean_b\left(\frac{F_b(x_{b,k})-F_b^*}{F_b^*}\right)$',
                                fig_path = DATA_PATH, fig_name = 'optgap' )
#### ADDING MARKS AT THE X-AXIS WHERE THE PROXIMAL OPERATOR SOLVER STARTS TO FAIL
flag_dicts = pm.load_data( 'flag_hist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                           TAU_FISTA_UNTRAINED, FISTA_LD_LIST, DEEPOPT_LIST )
pm.add_marks( flag_dicts, optgap_fig_exp4, fig_path = DATA_PATH, fig_name = 'optgap_w_marks' )

######################### TRAINED ALGORITHMS' MEAN VALIDATION LOSS AND PENALTY ##################################
val_loss, penalty, metrics_label_list = pm.compute_metrics( TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM,
                                                            DATE, FISTA_LD_LIST, DEEPOPT_LIST )
pm.print_metrics_analysis( val_loss, penalty, metrics_label_list, table_path = DATA_PATH, table_name = 'metrics_table' )

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 4 (End) ------------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#######################################################################################################################
#################################################### EXPERIMENT 5 #####################################################
#######################################################################################################################

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 5 (Start) ----------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')
######################### TRAINING - TESTING PARAMETERS ##################################
TEST_MODE           = 'test'
DATASET             = 'mayo_clinic_512'
TEST_DATASET_RATIO  = 1.0
PROBLEM             = 'lstv'

# TRANING PARAMETERS
EPOCHS_TO_SAVE      = [ 20 ]
MINITER             = [  1 ]
MAXITER             = [ 20 ]
TRAIN_DATASET_RATIO = [ 1.0 ]

# FISTA PARAMETER
TAU_FISTA_UNTRAINED = [ 0.5 ]

# FISTA-LD PARAMETERS
ALPHA         = [ 10**(-3.0) ]
GAMMA         = [ -1.0 ]
CONST_TAU     = [ 0.5 ]
PARAMS        = list( product( list( zip( GAMMA, CONST_TAU ) ), ALPHA ) )
FISTA_LD_LIST = list( product( PARAMS, TRAIN_DATASET_RATIO, EPOCHS_TO_SAVE, list( zip( MINITER, MAXITER ) ) ) )

# DEEPOPT PARAMETER
ALPHA_DEEPOPT = []
DEEPOPT_LIST  = list( product( ALPHA_DEEPOPT, TRAIN_DATASET_RATIO, EPOCHS_TO_SAVE, list( zip( MINITER, MAXITER ) ) ) )

######################### DATA PATH ##################################
DATA_PATH = os.path.join(os.getcwd(), 'paper_results_data', 'exp5' )
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

######################### PLOTTING MEAN (F-F*)/F* ##################################
#### LOADING F DATA
F_dicts = pm.load_data( 'Fhist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                        TAU_FISTA_UNTRAINED, FISTA_LD_LIST, DEEPOPT_LIST )
#### COMPUTING MEAN (F-F*)/F*
mean_rel_opt_gap_dicts = pm.compute_mean_rel_opt_gap( F_dicts )
#### STACKING O(1/k**2)
iters = mean_rel_opt_gap_dicts[0].get('data').shape[0]
mean_rel_opt_gap_dicts = [ dict( data = 6000/(1 + np.arange(iters))**2, label = '$O(1/k^2)$', ls = ':', c = 'k' ) ] + mean_rel_opt_gap_dicts
#### STACKING THE X-COORDINATE
mean_rel_opt_gap_dicts = [ dict( data = 1 + np.arange(iters), label = '$k+1$', ls = None, c = None ) ] + mean_rel_opt_gap_dicts
#### PLOTTING
optgap_fig_exp5 = pm.plot_data( mean_rel_opt_gap_dicts,
                                ylabel = r'$\mean_b\left(\frac{F_b(x_{b,k})-F_b^*}{F_b^*}\right)$',
                                fig_path = DATA_PATH, fig_name = 'optgap' )
#### ADDING MARKS ON THE X-AXIS WHERE THE PROXIMAL OPERATOR SOLVER BEGINS TO FAIL
flag_dicts = pm.load_data( 'flag_hist', TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM, DATE,
                           TAU_FISTA_UNTRAINED, FISTA_LD_LIST, DEEPOPT_LIST )
pm.add_marks( flag_dicts, optgap_fig_exp5, fig_path = DATA_PATH, fig_name = 'optgap_w_marks' )

######################### TRAINED ALGORITHMS' MEAN VALIDATION LOSS AND PENALTY ##################################
val_loss, penalty, metrics_label_list = pm.compute_metrics( TEST_MODE, DATASET, TEST_DATASET_RATIO, PROBLEM,
                                                            DATE, FISTA_LD_LIST, DEEPOPT_LIST )
pm.print_metrics_analysis( val_loss, penalty, metrics_label_list, table_path = DATA_PATH, table_name = 'metrics_table' )

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 5 (End) ------------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

print( '\nAll data was saved to:', os.path.join(os.getcwd(), 'paper_results_data') )
