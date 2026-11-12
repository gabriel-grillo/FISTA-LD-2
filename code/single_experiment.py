import os
from datetime import datetime
### For reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["PYTHONHASHSEED"] = "42"

import tensorflow as tf

### GPU settings
TF_USE_GPU  = True
VRAM_MB = 13.0 * 1024
if TF_USE_GPU:
    # Listing visable GPUs and asserting that there is at least one
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        raise RuntimeError("No GPU detected.")
    # Configuring TensorFlow GPU memory limit for the first GPU
    tf.config.set_logical_device_configuration(
        gpus[0],
        [ tf.config.LogicalDeviceConfiguration( memory_limit = VRAM_MB) ]
    )
else:
    tf.config.set_visible_devices( [], 'GPU' )

### Tensorflow settings
# Setting random seeds for 'random', 'numpy' and 'tensorflow'
tf.keras.utils.set_random_seed(42)
# Enabling operations determinism
tf.config.experimental.enable_op_determinism()
# # Enabling debbuging to catch inf and nan
# tf.debugging.enable_check_numerics()

import time
from itertools import product
from traintest import test_untrained, train_fista_dev, test_fista_dev, train_deepopt, test_deepopt

#######################################################################################################
############################################## EDIT HERE ##############################################
#######################################################################################################

##### TRAINING - TESTING PARAMETERS
DATASET             = 'mayo_clinic_512'   # 'mayo_clinic_128' or 'mayo_clinic_512'
PROBLEM             = 'lstv'             # 'nnls' or 'lasso' or 'slasso' or 'nnslasso' or 'lstv'
DATE                = datetime.today().strftime('%Y-%m-%d')
# DATE                = '2026-11-07'

TRAIN_BATCH_SIZE    = 1
EPOCHS_TO_SAVE      = [ 20 ]
EPOCHS              = max(EPOCHS_TO_SAVE)
LOAD_SAVED_MODELS   = False
MINITER             = [  1 ]
MAXITER             = [ 10 ]

TEST_MODE           = 'val' # 'test' or 'val'
TEST_ITERS          = 500
TEST_BATCH_SIZE     = 32

TRAIN_DATASET_RATIO = [ 0.05 ]
TEST_DATASET_RATIO  = [ 0.05 ]
DATASET_RATIO = list( zip( TRAIN_DATASET_RATIO, TEST_DATASET_RATIO ) )

# FISTA PARAMETER
GAMMA_FISTA_UNTRAINED  = [ 1.1, 1.5, 2.0, 2.5 ]
ALPHA_FISTA_UNTRAINED  = [ 0.0, 0.25, 0.5, 0.75, 0.9 ]
T_VARIANT_UNTRAINED    = [ 1 ]
PARAMS_FISTA_UNTRAINED = list( product( GAMMA_FISTA_UNTRAINED, ALPHA_FISTA_UNTRAINED, T_VARIANT_UNTRAINED ) )

# FISTA-DEV PARAMETERS -- set a negative const_tau to enable varying tau
GAMMA     = [ ]
ALPHA     = [ ]
T_VARIANT = [ ]
PARAMS = list( product( GAMMA, ALPHA, T_VARIANT ) )
ABLATION = [ 'd', 'e', 'v', 'dev', 'theta', 'all' ]

# DEEPOPT PARAMETER
ALPHA_DO = [ ]

#######################################################################################################
############################################## EDIT HERE ##############################################
#######################################################################################################

##### For time measurement
start_time = time.time()

##### TEST UNTRAINED ALGORITHMS
for dataset_ratio in set(TEST_DATASET_RATIO):
    # print(f'Testing ISTA with {100*dataset_ratio} percent of data')
    # test_untrained( DATASET, PROBLEM, TEST_MODE, 'ista', TEST_BATCH_SIZE, TEST_ITERS, dataset_ratio = dataset_ratio )
    for params in PARAMS_FISTA_UNTRAINED:
        gamma, alpha, t_variant = params
        print(f'Testing FISTA ( gamma={gamma}, alpha={alpha}, t_variant={t_variant} ) with {100*dataset_ratio} percent of data')
        test_untrained( DATASET, PROBLEM, TEST_MODE, 'fista', TEST_BATCH_SIZE, TEST_ITERS,
                        gamma = gamma, alpha = alpha, t_variant = t_variant, dataset_ratio = dataset_ratio )

##### TRAINING - TESTING FISTA-DEV
for i, options in enumerate( product( DATASET_RATIO, list(zip(MINITER,MAXITER)), PARAMS, ABLATION ) ):
    dataset_ratio, iters, params, ablation = options
    train_dataset_ratio, test_dataset_ratio = dataset_ratio
    miniter, maxiter = iters
    gamma, alpha, t_variant = params

    label = 'FISTA-DEV ' + str(i+1) + ': '
    label += f'N=[{miniter},{maxiter}], '
    label += f'gamma={gamma}, '
    label += f'alpha={alpha}, '
    label += f't_variant={t_variant}, '
    label += f'data_ratio={train_dataset_ratio}, '
    label += f'ablation={ablation}.'

    print('Training ' + label)
    train_fista_dev( DATASET, PROBLEM, miniter, maxiter, gamma, alpha,
                     EPOCHS, TRAIN_BATCH_SIZE, DATE, LOAD_SAVED_MODELS, ablation, t_variant,
                     dataset_ratio = train_dataset_ratio, epochs_to_save = EPOCHS_TO_SAVE )
    
    for epochs in EPOCHS_TO_SAVE:
        print('Testing ' + label + f' trained with {epochs} epochs')
        test_fista_dev( DATASET, PROBLEM, TEST_MODE, miniter, maxiter, gamma, alpha, epochs,
                        TEST_BATCH_SIZE, TEST_ITERS, DATE, ablation, t_variant, dataset_ratio = dataset_ratio )

##### TRAINING - TESTING DEEPOPT
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

#### For time measurement
end_time = time.time()
print(f"\nElapsed time: {end_time - start_time:.5e} seconds")