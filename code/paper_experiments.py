import os
#### For reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["PYTHONHASHSEED"] = "42"

import pynvml
import tensorflow as tf

#### GPU settings
TF_USE_GPU  = True
MIN_VRAM_MB = 12 * 1024
if TF_USE_GPU:
    # Listing visable GPUs and asserting that there is at least one
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("No GPU detected.")
    # Asserting that the first GPU has sufficient free VRAM
    pynvml.nvmlInit()
    try:
        mem = pynvml.nvmlDeviceGetMemoryInfo( pynvml.nvmlDeviceGetHandleByIndex( 0 ) )
        if mem.free < MIN_VRAM_MB * 1024**2:
            raise RuntimeError(
                f"Insufficient free VRAM. "
                f"Required: {MIN_VRAM_MB / 1024:.1f} GiB. "
                f"Available: {mem.free / 1024**3:.1f} GiB."
            )
    finally:
        pynvml.nvmlShutdown()
    # Configuring TensorFlow GPU memory limit for the first GPU
    tf.config.set_logical_device_configuration(
        gpus[0],
        [ tf.config.LogicalDeviceConfiguration( memory_limit = MIN_VRAM_MB) ]
    )
else:
    tf.config.set_visible_devices( [], 'GPU' )

#### Tensorflow settings
# Setting random seeds for 'random', 'numpy' and 'tensorflow'
tf.keras.utils.set_random_seed(42)
# Enabling operations determinism
tf.config.experimental.enable_op_determinism()

import numpy as np
import time
from itertools import product
from traintest import test_untrained, train_fista_ld, test_fista_ld, train_deepopt, test_deepopt

#### For time measurement
start_time = time.time()

#### Date to save results
DATE = '0'

#######################################################################################################################
#################################################### EXPERIMENT 1 #####################################################
#######################################################################################################################

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 1 (Start) ----------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#### TRAINING - TESTING PARAMETERS
DATASET             = 'mayo_clinic_512'
PROBLEM             = 'lasso'
TRAIN_BATCH_SIZE    = 1
EPOCHS_TO_SAVE      = [ 20 ]
EPOCHS              = max(EPOCHS_TO_SAVE)
LOAD_SAVED_MODELS   = False
MINITER             = [  1 ]
MAXITER             = [ 20 ]

TEST_MODE           = 'val'
TEST_ITERS          = 1000
TEST_BATCH_SIZE     = 64

TRAIN_DATASET_RATIO = [ 0.05 ]
TEST_DATASET_RATIO  = [ 1.0 ] * len(TRAIN_DATASET_RATIO)
DATASET_RATIO = list( zip( TRAIN_DATASET_RATIO, TEST_DATASET_RATIO ) )

### TEST UNTRAINED ALGORITHMS
# FISTA PARAMETER
TAU_FISTA_UNTRAINED = [ 1.0 ]
for dataset_ratio in set(TEST_DATASET_RATIO):
    print(f'Testing ISTA with {100*dataset_ratio} percent of data')
    test_untrained( DATASET, PROBLEM, TEST_MODE, 'ista', TEST_BATCH_SIZE, TEST_ITERS, dataset_ratio = dataset_ratio )
    for tau_fista in TAU_FISTA_UNTRAINED:
        print(f'Testing FISTA (tau={tau_fista}) with {100*dataset_ratio} percent of data')
        test_untrained( DATASET, PROBLEM, TEST_MODE, 'fista', TEST_BATCH_SIZE, TEST_ITERS, tau = tau_fista, dataset_ratio = dataset_ratio )

### TRAINING - TESTING FISTA-LD -- set a negative const_tau to enable varying tau
ALPHA       = np.logspace(-3.5, -1.5, 5)
GAMMA       = [ 0.01, 0.025, 0.05, 0.075, 0.1 ]
CONST_TAU   = [ -1.0, -1.0, -1.0, -1.0, -1.0 ]
PARAMS      = list( product( ALPHA, list( zip( GAMMA, CONST_TAU ) ) ) )
for i, options in enumerate( product( DATASET_RATIO, list(zip(MINITER,MAXITER)), PARAMS ) ):
    dataset_ratio, iters, params = options
    train_dataset_ratio, test_dataset_ratio = dataset_ratio
    alpha, gamma_const_tau = params
    gamma, const_tau = gamma_const_tau
    miniter, maxiter = iters

    label = 'FISTA-LD ' + str(i+1) + ': '
    label += f'N=[{miniter},{maxiter}], '
    label += f'alpha={alpha}, '
    if const_tau <= 0:
        label += f'gamma={gamma}, '
    else:
        label += f'tau={const_tau}, '
    label += f'data_ratio={train_dataset_ratio}'

    print('Training ' + label)
    train_fista_ld( DATASET, PROBLEM, miniter, maxiter, alpha, gamma, const_tau,
                    EPOCHS, TRAIN_BATCH_SIZE, DATE, LOAD_SAVED_MODELS,
                    dataset_ratio = train_dataset_ratio, epochs_to_save = EPOCHS_TO_SAVE )
    
    for epochs in EPOCHS_TO_SAVE:
        print('Testing ' + label + f' trained with {epochs} epochs')
        test_fista_ld( DATASET, PROBLEM, TEST_MODE, miniter, maxiter, alpha, gamma, const_tau, epochs,
                       TEST_BATCH_SIZE, TEST_ITERS, DATE, dataset_ratio = dataset_ratio )

#### TRAINING - TESTING DEEPOPT
ALPHA_DO = [ 0.1, 0.25, 0.5, 0.75, 0.9 ]
for i, options in enumerate( product( DATASET_RATIO, list(zip(MINITER,MAXITER)), ALPHA_DO ) ):
    dataset_ratio, iters, alpha = options
    train_dataset_ratio, test_dataset_ratio = dataset_ratio
    miniter, maxiter = iters

    label = 'DeepOpt ' + str(i+1) + ': '
    label += f'N=[{miniter},{maxiter}], '
    label += f'alpha={alpha}, '
    label += f'data_ratio={train_dataset_ratio}'

    print('Training ' + label)
    train_deepopt( DATASET, PROBLEM, miniter, maxiter, alpha,
                   EPOCHS, TRAIN_BATCH_SIZE, DATE, LOAD_SAVED_MODELS,
                   dataset_ratio = train_dataset_ratio, epochs_to_save = EPOCHS_TO_SAVE )
    
    for epochs in EPOCHS_TO_SAVE:
        print('Testing ' + label + f' trained with {epochs} epochs')
        test_deepopt( DATASET, PROBLEM, TEST_MODE, miniter, maxiter, alpha, epochs, DATE,
                      TEST_BATCH_SIZE, TEST_ITERS, dataset_ratio = dataset_ratio )

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 1 (End) ------------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#######################################################################################################################
#################################################### EXPERIMENT 2 #####################################################
#######################################################################################################################

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 2 (Start) ----------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#### TRAINING - TESTING PARAMETERS
DATASET             = 'mayo_clinic_512'
PROBLEM             = 'lasso'
TRAIN_BATCH_SIZE    = 1
EPOCHS_TO_SAVE      = [ 20, 40, 60, 80, 100 ]
EPOCHS              = max(EPOCHS_TO_SAVE)
LOAD_SAVED_MODELS   = False
MINITER             = [  1 ]
MAXITER             = [ 20 ]

TEST_MODE           = 'val'
TEST_ITERS          = 1000
TEST_BATCH_SIZE     = 64

TRAIN_DATASET_RATIO = [ 1.0 ]
TEST_DATASET_RATIO  = [ 1.0 ] * len(TRAIN_DATASET_RATIO)
DATASET_RATIO = list( zip( TRAIN_DATASET_RATIO, TEST_DATASET_RATIO ) )

#### TEST UNTRAINED ALGORITHMS
# same as Experiment 1

### TRAINING - TESTING FISTA-LD -- set a negative const_tau to enable varying tau
ALPHA     = [ 10**(-3.0) ]
GAMMA     = [ 0.025 ]
CONST_TAU = [ -1.0 ]
PARAMS    = list( zip( ALPHA, list( zip( GAMMA, CONST_TAU ) ) ) )
for i, options in enumerate( product( DATASET_RATIO, list(zip(MINITER,MAXITER)), PARAMS ) ):
    dataset_ratio, iters, params = options
    train_dataset_ratio, test_dataset_ratio = dataset_ratio
    alpha, gamma_const_tau = params
    gamma, const_tau = gamma_const_tau
    miniter, maxiter = iters

    label = 'FISTA-LD ' + str(i+1) + ': '
    label += f'N=[{miniter},{maxiter}], '
    label += f'alpha={alpha}, '
    if const_tau <= 0:
        label += f'gamma={gamma}, '
    else:
        label += f'tau={const_tau}, '
    label += f'data_ratio={train_dataset_ratio}'

    print('Training ' + label)
    train_fista_ld( DATASET, PROBLEM, miniter, maxiter, alpha, gamma, const_tau,
                    EPOCHS, TRAIN_BATCH_SIZE, DATE, LOAD_SAVED_MODELS,
                    dataset_ratio = train_dataset_ratio, epochs_to_save = EPOCHS_TO_SAVE )
    
    for epochs in EPOCHS_TO_SAVE:
        print('Testing ' + label + f' trained with {epochs} epochs')
        test_fista_ld( DATASET, PROBLEM, TEST_MODE, miniter, maxiter, alpha, gamma, const_tau, epochs,
                       TEST_BATCH_SIZE, TEST_ITERS, DATE, dataset_ratio = dataset_ratio )
        
print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 2 (End) ------------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#######################################################################################################################
#################################################### EXPERIMENT 3 #####################################################
#######################################################################################################################

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 3 (Start) ----------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#### TRAINING - TESTING PARAMETERS
DATASET             = 'mayo_clinic_512'
PROBLEM             = 'lasso'
TRAIN_BATCH_SIZE    = 1
EPOCHS_TO_SAVE      = [ 20 ]
EPOCHS              = max(EPOCHS_TO_SAVE)
LOAD_SAVED_MODELS   = False
MINITER             = [  1 ]
MAXITER             = [ 20 ]

TEST_MODE           = 'test'
TEST_ITERS          = 1000
TEST_BATCH_SIZE     = 64

TRAIN_DATASET_RATIO = [ 1.0 ]
TEST_DATASET_RATIO  = [ 1.0 ] * len(TRAIN_DATASET_RATIO)
DATASET_RATIO = list( zip( TRAIN_DATASET_RATIO, TEST_DATASET_RATIO ) )

#### TEST UNTRAINED ALGORITHMS
# FISTA PARAMETER
TAU_FISTA_UNTRAINED = [ 1.0 ]
for dataset_ratio in set(TEST_DATASET_RATIO):
    print(f'Testing ISTA with {100*dataset_ratio} percent of data')
    test_untrained( DATASET, PROBLEM, TEST_MODE, 'ista', TEST_BATCH_SIZE, TEST_ITERS, dataset_ratio = dataset_ratio )
    for tau_fista in TAU_FISTA_UNTRAINED:
        print(f'Testing FISTA (tau={tau_fista}) with {100*dataset_ratio} percent of data')
        test_untrained( DATASET, PROBLEM, TEST_MODE, 'fista', TEST_BATCH_SIZE, TEST_ITERS, tau = tau_fista, dataset_ratio = dataset_ratio )

### TRAINING - TESTING FISTA-LD -- set a negative const_tau to enable varying tau
ALPHA     = [ 10**(-3.0) ]
GAMMA     = [ 0.025 ]
CONST_TAU = [ -1.0 ]
PARAMS    = list( zip( ALPHA, list( zip( GAMMA, CONST_TAU ) ) ) )
for i, options in enumerate( product( DATASET_RATIO, list(zip(MINITER,MAXITER)), PARAMS ) ):
    dataset_ratio, iters, params = options
    train_dataset_ratio, test_dataset_ratio = dataset_ratio
    alpha, gamma_const_tau = params
    gamma, const_tau = gamma_const_tau
    miniter, maxiter = iters

    label = 'FISTA-LD ' + str(i+1) + ': '
    label += f'N=[{miniter},{maxiter}], '
    label += f'alpha={alpha}, '
    if const_tau <= 0:
        label += f'gamma={gamma}, '
    else:
        label += f'tau={const_tau}, '
    label += f'data_ratio={train_dataset_ratio}'

    # Only need to test (training was performed in experiment 2)    
    for epochs in EPOCHS_TO_SAVE:
        print('Testing ' + label + f' trained with {epochs} epochs')
        test_fista_ld( DATASET, PROBLEM, TEST_MODE, miniter, maxiter, alpha, gamma, const_tau, epochs,
                       TEST_BATCH_SIZE, TEST_ITERS, DATE, dataset_ratio = dataset_ratio )

### TRAINING - TESTING DEEPOPT
ALPHA_DO = [ 0.5 ]
for i, options in enumerate( product( DATASET_RATIO, list(zip(MINITER,MAXITER)), ALPHA_DO ) ):
    dataset_ratio, iters, alpha = options
    train_dataset_ratio, test_dataset_ratio = dataset_ratio
    miniter, maxiter = iters

    label = 'DeepOpt ' + str(i+1) + ': '
    label += f'N=[{miniter},{maxiter}], '
    label += f'alpha={alpha}, '
    label += f'data_ratio={train_dataset_ratio}'

    print('Training ' + label)
    train_deepopt( DATASET, PROBLEM, miniter, maxiter, alpha,
                   EPOCHS, TRAIN_BATCH_SIZE, DATE, LOAD_SAVED_MODELS,
                   dataset_ratio = train_dataset_ratio, epochs_to_save = EPOCHS_TO_SAVE )
    
    for epochs in EPOCHS_TO_SAVE:
        print('Testing ' + label + f' trained with {epochs} epochs')
        test_deepopt( DATASET, PROBLEM, TEST_MODE, miniter, maxiter, alpha, epochs, DATE,
                      TEST_BATCH_SIZE, TEST_ITERS, dataset_ratio = dataset_ratio )
        
print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 3 (End) ------------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#######################################################################################################################
#################################################### EXPERIMENT 4 #####################################################
#######################################################################################################################

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 4 (Start) ----------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#### TRAINING - TESTING PARAMETERS
DATASET             = 'mayo_clinic_512'
PROBLEM             = 'lstv'
TRAIN_BATCH_SIZE    = 1
EPOCHS_TO_SAVE      = [ 20 ]
EPOCHS              = max(EPOCHS_TO_SAVE)
LOAD_SAVED_MODELS   = False
MINITER             = [  1 ]
MAXITER             = [ 20 ]

TEST_MODE           = 'val'
TEST_ITERS          = 200
TEST_BATCH_SIZE     = 64

TRAIN_DATASET_RATIO = [ 0.05 ]
TEST_DATASET_RATIO  = [ 1.0 ] * len(TRAIN_DATASET_RATIO)
DATASET_RATIO = list( zip( TRAIN_DATASET_RATIO, TEST_DATASET_RATIO ) )

#### TEST UNTRAINED ALGORITHMS
# FISTA PARAMETER
TAU_FISTA_UNTRAINED = [ 0.1, 0.25, 0.5, 0.75, 0.9 ]
for dataset_ratio in set(TEST_DATASET_RATIO):
    for tau_fista in TAU_FISTA_UNTRAINED:
        print(f'Testing FISTA (tau={tau_fista}) with {100*dataset_ratio} percent of data')
        test_untrained( DATASET, PROBLEM, TEST_MODE, 'fista', TEST_BATCH_SIZE, TEST_ITERS, tau = tau_fista, dataset_ratio = dataset_ratio )

### TRAINING - TESTING FISTA-LD
ALPHA     = [ 10**(-3.0) ]
GAMMA     = [ -1.0, -1.0, -1.0, -1.0, -1.0 ]
CONST_TAU = [ 0.1, 0.25, 0.5, 0.75, 0.9 ]
PARAMS      = list( product( ALPHA, list( zip( GAMMA, CONST_TAU ) ) ) )
for i, options in enumerate( product( DATASET_RATIO, list(zip(MINITER,MAXITER)), PARAMS ) ):
    dataset_ratio, iters, params = options
    train_dataset_ratio, test_dataset_ratio = dataset_ratio
    alpha, gamma_const_tau = params
    gamma, const_tau = gamma_const_tau
    miniter, maxiter = iters

    label = 'FISTA-LD ' + str(i+1) + ': '
    label += f'N=[{miniter},{maxiter}], '
    label += f'alpha={alpha}, '
    if const_tau <= 0:
        label += f'gamma={gamma}, '
    else:
        label += f'tau={const_tau}, '
    label += f'data_ratio={train_dataset_ratio}'

    print('Training ' + label)
    train_fista_ld( DATASET, PROBLEM, miniter, maxiter, alpha, gamma, const_tau,
                    EPOCHS, TRAIN_BATCH_SIZE, DATE, LOAD_SAVED_MODELS,
                    dataset_ratio = train_dataset_ratio, epochs_to_save = EPOCHS_TO_SAVE )
    
    for epochs in EPOCHS_TO_SAVE:
        print('Testing ' + label + f' trained with {epochs} epochs')
        test_fista_ld( DATASET, PROBLEM, TEST_MODE, miniter, maxiter, alpha, gamma, const_tau, epochs,
                       TEST_BATCH_SIZE, TEST_ITERS, DATE, dataset_ratio = dataset_ratio )
        
print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 4 (End) ------------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#######################################################################################################################
#################################################### EXPERIMENT 5 #####################################################
#######################################################################################################################

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 5 (Start) ----------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#### TRAINING - TESTING PARAMETERS
DATASET             = 'mayo_clinic_512'
PROBLEM             = 'lstv'
TRAIN_BATCH_SIZE    = 1
EPOCHS_TO_SAVE      = [ 20 ]
EPOCHS              = max(EPOCHS_TO_SAVE)
LOAD_SAVED_MODELS   = False
MINITER             = [  1 ]
MAXITER             = [ 20 ]

TEST_MODE           = 'test'
TEST_ITERS          = 200
TEST_BATCH_SIZE     = 64

TRAIN_DATASET_RATIO = [ 1.0 ]
TEST_DATASET_RATIO  = [ 1.0 ] * len(TRAIN_DATASET_RATIO)
DATASET_RATIO = list( zip( TRAIN_DATASET_RATIO, TEST_DATASET_RATIO ) )

#### TEST UNTRAINED ALGORITHMS
# FISTA PARAMETER
TAU_FISTA_UNTRAINED = [ 0.5 ]
for dataset_ratio in set(TEST_DATASET_RATIO):
    for tau_fista in TAU_FISTA_UNTRAINED:
        print(f'Testing FISTA (tau={tau_fista}) with {100*dataset_ratio} percent of data')
        test_untrained( DATASET, PROBLEM, TEST_MODE, 'fista', TEST_BATCH_SIZE, TEST_ITERS, tau = tau_fista, dataset_ratio = dataset_ratio )

### TRAINING - TESTING FISTA-LD
ALPHA     = [ 10**(-3.0) ]
GAMMA     = [ -1.0 ]
CONST_TAU = [ 0.5 ]
PARAMS    = list( zip( ALPHA, list( zip( GAMMA, CONST_TAU ) ) ) )
for i, options in enumerate( product( DATASET_RATIO, list(zip(MINITER,MAXITER)), PARAMS ) ):
    dataset_ratio, iters, params = options
    train_dataset_ratio, test_dataset_ratio = dataset_ratio
    alpha, gamma_const_tau = params
    gamma, const_tau = gamma_const_tau
    miniter, maxiter = iters

    label = 'FISTA-LD ' + str(i+1) + ': '
    label += f'N=[{miniter},{maxiter}], '
    label += f'alpha={alpha}, '
    if const_tau <= 0:
        label += f'gamma={gamma}, '
    else:
        label += f'tau={const_tau}, '
    label += f'data_ratio={train_dataset_ratio}'

    print('Training ' + label)
    train_fista_ld( DATASET, PROBLEM, miniter, maxiter, alpha, gamma, const_tau,
                    EPOCHS, TRAIN_BATCH_SIZE, DATE, LOAD_SAVED_MODELS,
                    dataset_ratio = train_dataset_ratio, epochs_to_save = EPOCHS_TO_SAVE )
    
    for epochs in EPOCHS_TO_SAVE:
        print('Testing ' + label + f' trained with {epochs} epochs')
        test_fista_ld( DATASET, PROBLEM, TEST_MODE, miniter, maxiter, alpha, gamma, const_tau, epochs,
                       TEST_BATCH_SIZE, TEST_ITERS, DATE, dataset_ratio = dataset_ratio )

print('\n------------------------------------------------------------------------------------------------------')
print('---------------------------------------- EXPERIMENT 5 (End) ------------------------------------------')
print('------------------------------------------------------------------------------------------------------\n')

#### For time measurement
end_time = time.time()
print(f"\nElapsed time: {end_time - start_time:.5e} seconds")
