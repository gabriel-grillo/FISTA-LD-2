import os
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as pp
pp.rcParams[ 'savefig.bbox' ] = 'tight'

from operators import reconstruction_operators, torch_linear_op_as_tf2_layer, wavelet_as_tf2_layer
from data import get_dataset, get_observed_data
from optimization import fista, ista, fista_ld, deepopt_nonsmooth
from paths import model_path, hist_path

RANDOM_SEED = 42
SAVE_RECONSTRUCTIONS = False
PRINT_FINAL_LOSS = True

#######################################################################################################################
################################################ PROBLEM'S DEFINITION #################################################
#######################################################################################################################
def problem_definition( dataset, mode, problem, batch_size, dataset_ratio ):
    ##### OPERATORS FOR RECONSTRUCTION (OPTIMIZATION)
    ray_trafo, ray_trafo_transp, img_shape, sino_shape, \
        ray_trafo_norm, reg_trafo, reg_trafo_transp, reg_trafo_norm, lam = reconstruction_operators( dataset, problem )
    
    ##### DOMAIN SHAPE
    x_shape = img_shape + (1,) # one channel
    
    ##### TENSORFLOW INTEGRATION
    A = torch_linear_op_as_tf2_layer( ray_trafo, ray_trafo_transp, img_shape, sino_shape, np.float32, name = 'ray_transform' )
    A_adj = torch_linear_op_as_tf2_layer( ray_trafo_transp, ray_trafo, sino_shape, img_shape, np.float32, name = 'ray_transform_adj' )
    A_norm = tf.constant( ray_trafo_norm, dtype = tf.float32, name = 'A_norm' )
    if problem == 'lasso':
        W = wavelet_as_tf2_layer( reg_trafo, reg_trafo_transp, img_shape, np.float32, name = 'regularization_transform' )
        W_adj = wavelet_as_tf2_layer( reg_trafo_transp, reg_trafo, img_shape, np.float32, name = 'regularization_transform_adj' )
        lam = tf.constant( lam[None,...,None], dtype = tf.float32, name = 'lam' )
    else:
        W = reg_trafo
        W_adj = reg_trafo_transp
        lam = tf.constant( lam, dtype = tf.float32, name = 'lam' )
    W_norm = tf.constant( reg_trafo_norm, dtype = tf.float32, name = 'W_norm' )
    
    if mode == 'train':
        ###### REFERENCE FUNCTIONAL VALUES FOR TRAINING
        gamma_FISTA   = 1.5
        alpha_FISTA   = 0.7
        if problem == 'lstv':
            maxiter_FISTA = 200
        else:
            maxiter_FISTA = 400
        @tf.function
        def F_ref( b ):
            # Initial point
            x0  = tf.zeros( shape = (tf.shape(b)[0],) + x_shape, dtype = tf.float32, name = 'x0' )
            # Iterates FISTA saving functional values
            _, _, hist = fista( problem, x0, A, A_adj, A_norm, b, W, W_adj, W_norm, lam,
                                maxiter_FISTA, gamma_FISTA, alpha_FISTA,
                                save_hist = True
                            )
            # Returns only the functional values history
            return hist[0]
        
        ##### DATASET
        ds = get_dataset( dataset, mode, problem, batch_size, F_ref, overall_ratio = dataset_ratio )
    else:
        ##### DATASET
        ds = get_observed_data( dataset, mode, batch_size, overall_ratio = dataset_ratio )

    return ds, A, A_adj, A_norm, W, W_adj, W_norm, lam, x_shape

#######################################################################################################################
################################################### TRAIN ROUTINES ####################################################
#######################################################################################################################

def training_loop( epochs, train_step, ds_train, save_weights, saving_path, load_saved_info = False, epochs_to_save = [] ):
    ##### LOADING TRAINING INFO
    if load_saved_info:
        with open( os.path.join( saving_path, 'info.dat' ), 'r' ) as file:
            lines = np.loadtxt( file )
            initial_epoch = int(lines[0])
        with open( os.path.join( saving_path, 'hist.dat' ), 'r' ) as file:
            hist = np.loadtxt( file )
            hist = np.concatenate( (hist, np.zeros((2,epochs))), axis = 1 )
            train_loss_hist = hist[0]
            time_hist = hist[1]
    else:
        initial_epoch = 0
        train_loss_hist = np.zeros(epochs)
        time_hist = np.zeros(epochs)

    ##### TRAINING LOOP
    for epoch in range(initial_epoch,initial_epoch+epochs):
        start_time = time.time()

        # Iterate over the batches of the dataset.
        mean_train_loss = 0.0
        for ( b_train, F_b ) in ds_train:
            # Training step
            loss = train_step( b_train, F_b )
            # Mean train loss accumulation
            mean_train_loss += loss.numpy()
        mean_train_loss /= float( ds_train.cardinality().numpy() )
        train_loss_hist[epoch] = mean_train_loss

        time_hist[epoch] = time.time() - start_time

        # Printing new info
        print( '--------------------------------------------------------------------------------' )
        print( "Results of epoch %d" % (epoch,) )
        print( "Mean train loss at epoch %d: %.4f" % ( epoch, mean_train_loss ) )
        print( "Time taken: %.2fs" % ( time_hist[epoch] ) )

        # Saving the weights every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) in epochs_to_save:
            save_weights(epoch + 1)
    # Saving final weights
    save_weights(epoch + 1)

    if epochs > 0:
        ##### PRINTING FINAL TRAINING INFO
        print('--------------------------------------------------------------------------------')
        print('--------------------------------------------------------------------------------')
        print( f'Total number of epochs: {epoch+1}' )
        print( f'Average training time per epoch (s): { np.sum(time_hist[1:]) / (epochs-1) }' )
        print('--------------------------------------------------------------------------------')
        print('--------------------------------------------------------------------------------')
        ##### PLOTTING HIST INFO
        pp.semilogy(train_loss_hist[:epoch+1], marker = 'o', label = 'Train loss')
        pp.grid()
        pp.legend()
        pp.savefig( os.path.join( saving_path, 'train_hist.png' ) )
        pp.close()
        ##### SAVING FINAL TRAINING INFO
        if os.path.exists( os.path.join( saving_path, 'hist.dat' ) ):
            os.remove( os.path.join( saving_path, 'hist.dat' ) )
        with open( os.path.join( saving_path, 'hist.dat' ), 'ab' ) as file:
            hist = np.vstack( (train_loss_hist, time_hist) )
            np.savetxt( file, hist[:,:epoch+1] )
        if os.path.exists( os.path.join( saving_path, 'info.dat' ) ):
            os.remove( os.path.join( saving_path, 'info.dat' ) )
        with open( os.path.join( saving_path, 'info.dat' ), 'ab' ) as file:
            np.savetxt( file, [epoch+1] )
            np.savetxt( file, [np.sum(time_hist[1:]) / (epochs-1)] )

def network_base_arch( input_list, name ):
    FILTERS = 32
    KSIZE = 3
    N_OUT = 1
    N_LAYERS = 2

    input_convnet = keras.layers.Concatenate(axis=-1)(input_list)
    x = keras.layers.GroupNormalization(groups=-1)(input_convnet)
    for _ in range(N_LAYERS):
        x = keras.layers.Conv2D(FILTERS, KSIZE,
                                padding='same',
                                kernel_regularizer=keras.regularizers.L2(1e-3),
                                kernel_initializer=keras.initializers.VarianceScaling(scale = 1e-2, seed=RANDOM_SEED))(x)
        x = keras.layers.GroupNormalization(groups=-1)(x)
        x = keras.layers.LeakyReLU()(x)
    result = keras.layers.Conv2D(N_OUT, KSIZE,
                                 padding='same',
                                 kernel_regularizer=keras.regularizers.L2(1e-3),
                                 kernel_initializer=keras.initializers.VarianceScaling(scale = 1e-2, seed=RANDOM_SEED))(x)

    return keras.Model( inputs = input_list, outputs = result, name = name )

def train_fista_ld( dataset, problem, miniter, maxiter, gamma, alpha,
                    epochs, batch_size, date, load_saved_models, 
                    dataset_ratio = 1.0, epochs_to_save = [] ):
    ######################### PROBLEM'S DEFINITION ##################################
    ds_train, A, A_adj, A_norm, W, W_adj, W_norm, lam, x_shape = \
        problem_definition( dataset, 'train', problem, batch_size, dataset_ratio )

    ######################### MODELS' PATH ##################################
    MODEL_PATH = model_path( 'fista-ld',
                             [dataset, dataset_ratio, problem, miniter, maxiter, gamma, alpha, date],
                             load_saved_models,
                             training = True )

    ######################### MODELS' DEFINITION ##################################
    ##### D MODEL
    if load_saved_models:
        with open( os.path.join( MODEL_PATH, 'info.dat' ), 'r' ) as file:
            lines = np.loadtxt( file )
            last_epoch = int(lines[0])
        d_model = keras.models.load_model( os.path.join( MODEL_PATH, f'epoch_{int(last_epoch)}', 'd_model.keras' ) )
    else:
        input_y      = keras.Input(shape=x_shape, name='y')
        input_grad   = keras.Input(shape=x_shape, name='grad')
        input_d_prev = keras.Input(shape=x_shape, name='d_prev')
        d_input_list = [ input_y, input_grad, input_d_prev ]
        d_model      = network_base_arch( d_input_list, "d_model" )
    d_model.summary()
    ##### E MODEL
    if load_saved_models:
        with open( os.path.join( MODEL_PATH, 'info.dat' ), 'r' ) as file:
            lines = np.loadtxt( file )
            last_epoch = int(lines[0])
        e_model = keras.models.load_model( os.path.join( MODEL_PATH, f'epoch_{int(last_epoch)}', 'e_model.keras' ) )
    else:
        input_y              = keras.Input(shape=x_shape, name='y')
        input_prox_direction = keras.Input(shape=x_shape, name='prox_direction')
        input_d              = keras.Input(shape=x_shape, name='d')
        input_e_prev         = keras.Input(shape=x_shape, name='e_prev')
        e_input_list = [ input_y, input_prox_direction, input_d, input_e_prev ]
        e_model      = network_base_arch( e_input_list, "e_model" )
    e_model.summary()
    ##### V MODEL
    if load_saved_models:
        with open( os.path.join( MODEL_PATH, 'info.dat' ), 'r' ) as file:
            lines = np.loadtxt( file )
            last_epoch = int(lines[0])
        v_model = keras.models.load_model( os.path.join( MODEL_PATH, f'epoch_{int(last_epoch)}', 'v_model.keras' ) )
    else:
        input_x                  = keras.Input(shape=x_shape, name='x')
        input_inertial_direction = keras.Input(shape=x_shape, name='inertial_direction')
        input_d              = keras.Input(shape=x_shape, name='d')
        input_e              = keras.Input(shape=x_shape, name='e')
        input_v_prev             = keras.Input(shape=x_shape, name='v_prev')
        v_input_list = [ input_x, input_inertial_direction, input_d, input_e, input_v_prev ]
        v_model      = network_base_arch( v_input_list, "v_model" )
    v_model.summary()
    ##### TRAINABLE STEPSIZES
    theta = tf.Variable( initial_value = tf.zeros(maxiter), dtype = tf.float32, trainable = True, name='theta' )
    ##### SAVING INITIALIZED MODELS
    def save_weights(epoch):
        path_epoch = os.path.join( MODEL_PATH, f'epoch_{epoch}')
        if not os.path.exists( path_epoch ):
            os.makedirs( path_epoch )
        d_model.save( os.path.join( path_epoch, 'd_model.keras' ) )
        e_model.save( os.path.join( path_epoch, 'e_model.keras' ) )
        v_model.save( os.path.join( path_epoch, 'v_model.keras' ) )
        np.save( os.path.join( path_epoch, 'theta.npy' ), theta.numpy() )
    # Always saves the initialized model
    if not load_saved_models:
        save_weights(0)

    ######################### TRAINING ##################################
    ##### CHOOSE OPTIMIZER
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    ##### TRAINING STEP
    @tf.function
    def train_step( b_train, F_ref ):
        # Initial point
        x0_train  = tf.zeros( shape = (tf.shape(b_train)[0],) + x_shape, dtype = tf.float32, name = 'x0' )
        # Number of iterations
        if miniter == maxiter:
            n_iter_train = tf.constant( maxiter, dtype = tf.int32, name = 'maxiter' )
        else:
            n_iter_train = tf.random.uniform( shape = [],  minval = miniter, maxval = maxiter + 1, \
                                              dtype = tf.int32, name = 'maxiter', seed = RANDOM_SEED )
        with tf.GradientTape() as tape:
            _, F, _ = fista_ld( problem, x0_train, A, A_adj, A_norm, b_train, W, W_adj, W_norm, lam,
                                d_model, e_model, v_model, theta,
                                n_iter_train, n_iter_train, gamma, alpha,
                                training = True, save_hist = False
                                )
            loss = tf.reduce_mean( ( F - F_ref[:,-1] ) / ( F_ref[:,n_iter_train] - F_ref[:,-1] ) )
            reg_loss = tf.add_n( d_model.losses + e_model.losses + v_model.losses )
            total_loss = loss + reg_loss
        variables = d_model.trainable_variables + e_model.trainable_variables + v_model.trainable_variables + [ theta ]
        grads = tape.gradient( total_loss, variables )
        optimizer.apply_gradients( zip( grads, variables ) )
        return loss
    ##### TRAINING LOOP
    training_loop( epochs, train_step, ds_train, save_weights, MODEL_PATH, load_saved_info = load_saved_models, epochs_to_save = epochs_to_save )


def train_deepopt( dataset, problem, miniter, maxiter, alpha,
                   epochs, batch_size, date, load_saved_models,
                   dataset_ratio = 1.0, epochs_to_save = [] ):
    ######################### PROBLEM'S DEFINITION ##################################
    ds_train, A, A_adj, A_norm, W, W_adj, W_norm, lam, x_shape = \
        problem_definition( dataset, 'train', problem, batch_size, dataset_ratio )
    
    ######################### MODELS' PATH ##################################
    MODEL_PATH = model_path( 'deepopt',
                             [dataset, dataset_ratio, problem, miniter, maxiter, alpha, date],
                             load_saved_models,
                             training = True )

    ######################### MODELS' DEFINITION ##################################
    ##### DX1 MODEL
    if load_saved_models:
        with open( os.path.join( MODEL_PATH, 'info.dat' ), 'r' ) as file:
            lines = np.loadtxt( file )
            last_epoch = int(lines[0])
        dx1_model = keras.models.load_model( os.path.join( MODEL_PATH, f'epoch_{int(last_epoch)}', 'dx1.keras' ) )
    else:
        input_x        = keras.Input(shape=x_shape, name='x')
        input_grad     = keras.Input(shape=x_shape, name='grad')
        input_dx1      = keras.Input(shape=x_shape, name='dx1')
        dx1_input_list = [input_x, input_grad, input_dx1]
        dx1_model      = network_base_arch( dx1_input_list, "dx1_model" )
    dx1_model.summary()
    ##### DX2 MODEL
    if load_saved_models:
        with open( os.path.join( MODEL_PATH, 'info.dat' ), 'r' ) as file:
            lines = np.loadtxt( file )
            last_epoch = int(lines[0])
        dx2_model = keras.models.load_model( os.path.join( MODEL_PATH, f'epoch_{int(last_epoch)}', 'dx2.keras' ) )
    else:
        input_x        = keras.Input(shape=x_shape, name='x')
        input_grad     = keras.Input(shape=x_shape, name='grad')
        input_dx2      = keras.Input(shape=x_shape, name='dx2')
        input_dx1      = keras.Input(shape=x_shape, name='dx1')
        dx2_input_list = [input_x, input_grad, input_dx2, input_dx1]
        dx2_model      = network_base_arch( dx2_input_list, name = "dx2_model" )
    dx2_model.summary()
    ##### TRAINABLE STEPSIZES
    theta = tf.Variable( initial_value = tf.zeros(maxiter), dtype = tf.float32, trainable = True, name='theta' )
    ##### SAVING INITIALIZED MODELS
    def save_weights(epoch):
        if not os.path.exists( os.path.join( MODEL_PATH, f'epoch_{epoch}') ):
            os.makedirs( os.path.join( MODEL_PATH, f'epoch_{epoch}') )
        dx1_model.save( os.path.join( MODEL_PATH, f'epoch_{epoch}', 'dx1.keras' ) )
        dx2_model.save( os.path.join( MODEL_PATH, f'epoch_{epoch}', 'dx2.keras' ) )
        np.save( os.path.join( MODEL_PATH, f'epoch_{epoch}', 'theta.npy' ), theta.numpy() )
    # Always saves the initialized model
    if not load_saved_models:
        save_weights(0)

    ######################### TRAINING ##################################
    ##### CHOOSE OPTIMIZER
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    ##### TRAINING STEP
    @tf.function
    def train_step( b_train, F_ref ):
        # Initial point
        x0_train  = tf.zeros( shape = (tf.shape(b_train)[0],) + x_shape, dtype = tf.float32, name = 'x0' )
        # Number of iterations
        if miniter == maxiter:
            n_iter_train = tf.constant( maxiter, dtype = tf.int32, name = 'maxiter' )
        else:
            n_iter_train = tf.random.uniform( shape = [],  minval = miniter, maxval = maxiter + 1, \
                                              dtype = tf.int32, name = 'maxiter', seed = RANDOM_SEED )
        # Open a GradientTape to record the operations run during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer. The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
            _, F, _ = deepopt_nonsmooth( problem, x0_train, A, A_adj, A_norm, b_train, W, W_adj, W_norm, lam, \
                                         dx1_model, dx2_model, theta, n_iter_train, n_iter_train, \
                                         tf.constant(alpha, dtype=tf.float32), \
                                         training = True, save_hist = False)
            loss = tf.reduce_mean( ( F - F_ref[:,-1] ) / ( F_ref[:,n_iter_train] - F_ref[:,-1] ) )
            reg_loss = tf.add_n( dx1_model.losses + dx2_model.losses )
            total_loss = loss + reg_loss
        # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
        variables = dx1_model.trainable_variables + dx2_model.trainable_variables + [ theta ]
        grads = tape.gradient(total_loss, variables)
        # Run one step of gradient descent by updating the value of the variables to minimize the loss.
        optimizer.apply_gradients( zip( grads, variables ) )
        return loss

    ##### TRAINING LOOP
    training_loop( epochs, train_step, ds_train, save_weights, MODEL_PATH, load_saved_info = load_saved_models, epochs_to_save = epochs_to_save )


#######################################################################################################################
################################################### TEST ROUTINES #####################################################
#######################################################################################################################
def test_loop( test_step, ds, batch_size, result_path, filenames ):
    hist_list = []
    for step, b in enumerate(ds):
        # Running the algorithm
        rec, hist = test_step( b )
        
        # Printing results
        for j in range(b.shape[0]):
            if PRINT_FINAL_LOSS:
                print("Test example ", step * batch_size + j, "------------------------")
                print("Final loss ", hist[0][j][-1].numpy())

            if SAVE_RECONSTRUCTIONS:
                # Saving reconstructions
                pp.imshow( rec[j,...,0], cmap = 'gray' )
                pp.yticks([])
                pp.xticks([])
                pp.ylabel(f'{tf.shape(rec[j,...,0])[0]}')
                pp.xlabel(f'{tf.shape(rec[j,...,0])[1]}')
                pp.colorbar()
                pp.savefig( os.path.join( result_path, 'image', 'rec' + str(step * batch_size + j) + '.png' ) )
                pp.close()

        # Saving test info to list
        hist_list.append( hist )

    # Saving hist info to file
    for i, fn in enumerate( filenames ):
        file_path = os.path.join( result_path, fn )
        if os.path.exists( file_path ):
            os.remove( file_path )
        with open( file_path, 'wb' ) as f:
            np.save( f, np.concatenate( [ h[i] for h in hist_list  ], axis = 0 ) )

def test_untrained( dataset, problem, mode, algorithm, batch_size, iters, gamma = None, alpha = None, dataset_ratio = 1.0 ):
    assert algorithm == 'fista' or algorithm == 'ista', 'Unimplemented algorithm: ' + algorithm
    ######################### PROBLEM'S DEFINITION ##################################
    ds, A, A_adj, A_norm, W, W_adj, W_norm, lam, x_shape = \
            problem_definition( dataset, mode, problem, batch_size, dataset_ratio )

    ######################### RESULTS' PATH ##################################
    if algorithm == 'fista':
        RESULT_PATH = hist_path( 'fista', [ mode, dataset, dataset_ratio, problem, gamma, alpha ] )
    else:
        RESULT_PATH = hist_path( 'ista', [ mode, dataset, dataset_ratio, problem ] )

    FILENAMES = [
        'Fhist.npy', 'eps_hist.npy', 'flag_hist.npy', 'inner_iter_hist.npy'
    ]

    ######################### TEST STEP ##################################
    if algorithm == 'fista':
        @tf.function
        def test_step( b ):
            # Initial point
            x0  = tf.zeros( shape = (tf.shape(b)[0],) + x_shape, dtype = tf.float32, name = 'x0' )
            # Number of iterations
            n_iter = tf.constant( iters, dtype = tf.int32, name = 'maxiter' )
            x, _, hist = fista( problem, x0, A, A_adj, A_norm, b, W, W_adj, W_norm, lam,
                                n_iter, gamma, alpha,
                                save_hist = True
                               )
            return x, hist
    else:
        @tf.function
        def test_step( b ):
            # Initial point
            x0  = tf.zeros( shape = (tf.shape(b)[0],) + x_shape, dtype = tf.float32, name = 'x0' )
            # Number of iterations
            n_iter = tf.constant( iters, dtype = tf.int32, name = 'maxiter' )
            x, _, hist = ista( problem, x0, A, A_adj, A_norm, b, W, W_adj, W_norm, lam,
                               n_iter, save_hist = True
                             )
            return x, hist
            
    ######################### RUNING OVER DATASET ##################################
    test_loop( test_step, ds, batch_size, RESULT_PATH, FILENAMES )


def test_fista_ld( dataset, problem, mode, miniter, maxiter, gamma, alpha,
                   epochs, batch_size, iters, date, dataset_ratio = (1.0,1.0) ):
    ######################### PROBLEM'S DEFINITION ##################################
    train_dataset_ratio, test_dataset_ratio = dataset_ratio
    ds, A, A_adj, A_norm, W, W_adj, W_norm, lam, x_shape = \
            problem_definition( dataset, mode, problem, batch_size, test_dataset_ratio )

    ######################### RESULTS' PATH ##################################
    RESULT_PATH = hist_path( 'fista-ld',
                             [ mode, dataset, train_dataset_ratio, test_dataset_ratio,
                               problem, miniter, maxiter, gamma, alpha, date, epochs ] )
    
    FILENAMES = [
        'Fhist.npy', 'eps_hist.npy', 'd_hist.npy', 'd_ub_hist.npy',
        'e_hist.npy', 'e_ub_hist.npy', 'v_hist.npy', 'v_ub_hist.npy', 'flag_hist.npy', 'inner_iter_hist.npy'
    ]

    ######################### TEST STEP ##################################
    ##### MODELS' PATH
    MODEL_PATH = model_path( 'fista-ld',
                             [dataset, train_dataset_ratio, problem, miniter, maxiter, gamma, alpha, date],
                             load_saved_models = True,
                             training = False )
    ##### LOADING THE MODELS
    MODEL_PATH = os.path.join( MODEL_PATH, f'epoch_{epochs}' )
    if os.path.exists(MODEL_PATH):
        d_model = keras.models.load_model( os.path.join( MODEL_PATH, 'd_model.keras') )
        e_model = keras.models.load_model( os.path.join( MODEL_PATH, 'e_model.keras') )
        v_model = keras.models.load_model( os.path.join( MODEL_PATH, 'v_model.keras') )
        theta   = tf.constant( np.load( os.path.join( MODEL_PATH, 'theta.npy') ), dtype = tf.float32 )
    else:
        raise ValueError('This algorithm has not been trained')
    ##### TEST STEP
    @tf.function
    def test_step( b ):
        # Initial point
        x0  = tf.zeros( shape = (tf.shape(b)[0],) + x_shape, dtype = tf.float32, name = 'x0' )
        # Number of iterations
        n_iter = tf.constant( iters, dtype = tf.int32, name = 'maxiter' )
        x, _, hist = fista_ld( problem, x0, A, A_adj, A_norm, b, W, W_adj, W_norm, lam,
                               d_model, e_model, v_model, theta,
                               n_iter, maxiter, gamma, alpha,
                               training = False, save_hist = True
                              )
        return x, hist
    
    ######################### RUNING OVER DATASET ##################################
    test_loop( test_step, ds, batch_size, RESULT_PATH, FILENAMES )


def test_deepopt( dataset, problem, mode, miniter, maxiter, alpha, epochs, date,
                  batch_size, iters, dataset_ratio = (1.0,1.0) ):
    ######################### PROBLEM'S DEFINITION ##################################
    train_dataset_ratio, test_dataset_ratio = dataset_ratio
    ds, A, A_adj, A_norm, W, W_adj, W_norm, lam, x_shape = \
            problem_definition( dataset, mode, problem, batch_size, test_dataset_ratio )

    ######################### RESULTS' PATH ##################################
    RESULT_PATH = hist_path( 'deepopt',
                             [ mode, dataset, train_dataset_ratio, test_dataset_ratio,
                               problem, miniter, maxiter, alpha, date, epochs ] )
    FILENAMES = [
        'Fhist.npy', 'dx1hist.npy', 'dx1ubhist.npy', 'dx2hist.npy',
        'dx2ubhist.npy', 'flag_hist.npy', 'inner_iter_hist.npy'
    ]

    ######################### TEST STEP ##################################
    ##### MODELS' PATH
    MODEL_PATH = model_path( 'deepopt',
                             [dataset, train_dataset_ratio, problem, miniter, maxiter, alpha, date],
                             load_saved_models = True,
                             training = False )
    MODEL_PATH = os.path.join( MODEL_PATH, f'epoch_{epochs}' )
    ##### LOADING THE MODELS
    if os.path.exists(MODEL_PATH):
        dx1_model = keras.models.load_model( os.path.join( MODEL_PATH, 'dx1.keras') )
        dx2_model = keras.models.load_model( os.path.join( MODEL_PATH, 'dx2.keras') )
        theta     = tf.constant( np.load( os.path.join( MODEL_PATH, 'theta.npy') ), dtype = tf.float32 )
    else:
        raise ValueError('This algorithm has not been trained')       
    ##### TEST STEP
    @tf.function
    def test_step( b ):
        # Initial point
        x0  = tf.zeros( shape = (tf.shape(b)[0],) + x_shape, dtype = tf.float32, name = 'x0' )
        # Number of iterations
        n_iter = tf.constant( iters, dtype = tf.int32, name = 'maxiter' )
        x, _, hist = deepopt_nonsmooth( problem, x0, A, A_adj, A_norm, b, W, W_adj, W_norm, lam,
                                        dx1_model, dx2_model, theta, n_iter, maxiter, tf.constant(alpha, dtype=tf.float32),
                                        training = False, save_hist = True )
        return x, hist
    
    ######################### RUNING OVER DATASET ##################################
    test_loop( test_step, ds, batch_size, RESULT_PATH, FILENAMES )
