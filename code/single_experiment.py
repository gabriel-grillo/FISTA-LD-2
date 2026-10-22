import os
### For reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["PYTHONHASHSEED"] = "42"

# import pynvml
import tensorflow as tf

### GPU settings
TF_USE_GPU  = True
VRAM_MB = 3.5 * 1024
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
from traintest import test_untrained, train_fista_ld, test_fista_ld, train_deepopt, test_deepopt

#######################################################################################################
############################################## EDIT HERE ##############################################
#######################################################################################################

##### TRAINING - TESTING PARAMETERS
DATASET             = 'mayo_clinic_128'   # 'mayo_clinic_128' or 'mayo_clinic_512'
PROBLEM             = 'lasso'             # 'nnls' or 'lasso' or 'slasso' or 'nnslasso' or 'lstv'
DATE                = '2026-10-22'

TRAIN_BATCH_SIZE    = 1
EPOCHS_TO_SAVE      = [ 20 ]
EPOCHS              = max(EPOCHS_TO_SAVE)
LOAD_SAVED_MODELS   = False
MINITER             = [  2 ]
MAXITER             = [ 20 ]

TEST_MODE           = 'val' # 'test' or 'val'
TEST_ITERS          = 1000
TEST_BATCH_SIZE     = 32

TRAIN_DATASET_RATIO = [ 1.0 ]
TEST_DATASET_RATIO  = [ 1.0 ] * len(TRAIN_DATASET_RATIO)
DATASET_RATIO = list( zip( TRAIN_DATASET_RATIO, TEST_DATASET_RATIO ) )

# FISTA PARAMETER
GAMMA_FISTA_UNTRAINED  = [ 1.5, 1.5 ]
ALPHA_FISTA_UNTRAINED  = [ 0.3, 0.7 ]
PARAMS_FISTA_UNTRAINED = list( zip( GAMMA_FISTA_UNTRAINED, ALPHA_FISTA_UNTRAINED ) )

# FISTA-LD PARAMETERS -- set a negative const_tau to enable varying tau
GAMMA  = [ 1.5, 1.5 ]
ALPHA  = [ 0.3, 0.7 ]
PARAMS = list( zip( GAMMA, ALPHA ) )

# DEEPOPT PARAMETER
ALPHA_DO = [ 0.5 ]

#######################################################################################################
############################################## EDIT HERE ##############################################
#######################################################################################################

##### For time measurement
start_time = time.time()

# ##### TEST UNTRAINED ALGORITHMS
# for dataset_ratio in set(TEST_DATASET_RATIO):
#     print(f'Testing ISTA with {100*dataset_ratio} percent of data')
#     test_untrained( DATASET, PROBLEM, TEST_MODE, 'ista', TEST_BATCH_SIZE, TEST_ITERS, dataset_ratio = dataset_ratio )
#     for params in PARAMS_FISTA_UNTRAINED:
#         gamma, alpha = params
#         print(f'Testing FISTA ( gamma={gamma}, alpha={alpha} ) with {100*dataset_ratio} percent of data')
#         test_untrained( DATASET, PROBLEM, TEST_MODE, 'fista', TEST_BATCH_SIZE, TEST_ITERS, gamma = gamma, alpha = alpha, dataset_ratio = dataset_ratio )

##### TRAINING - TESTING FISTA-LD 
for i, options in enumerate( product( DATASET_RATIO, list(zip(MINITER,MAXITER)), PARAMS ) ):
    dataset_ratio, iters, params = options
    train_dataset_ratio, test_dataset_ratio = dataset_ratio
    miniter, maxiter = iters
    gamma, alpha = params

    label = 'FISTA-LD ' + str(i+1) + ': '
    label += f'N=[{miniter},{maxiter}], '
    label += f'gamma={gamma}, '
    label += f'alpha={alpha}, '
    label += f'data_ratio={train_dataset_ratio}.'

    print('Training ' + label)
    train_fista_ld( DATASET, PROBLEM, miniter, maxiter, gamma, alpha,
                    EPOCHS, TRAIN_BATCH_SIZE, DATE, LOAD_SAVED_MODELS,
                    dataset_ratio = train_dataset_ratio, epochs_to_save = EPOCHS_TO_SAVE )
    
    for epochs in EPOCHS_TO_SAVE:
        print('Testing ' + label + f' trained with {epochs} epochs')
        test_fista_ld( DATASET, PROBLEM, TEST_MODE, miniter, maxiter, gamma, alpha, epochs,
                       TEST_BATCH_SIZE, TEST_ITERS, DATE, dataset_ratio = dataset_ratio )

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