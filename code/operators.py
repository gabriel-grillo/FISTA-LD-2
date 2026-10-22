"""
    Buildind the operators involved in the problem
        min |Tx-b|_2^2 + lam * |Wx|_1
"""

import numpy as np
import tensorflow as tf
import torch
import ray_tracing_cuda_float_torch as rt
import pywt
from math import sqrt

RANDOM_SEED = 42

def power_method( A, A_T, domain_shape, maxiter = 1000, rtol = 1e-12, atol = 1e-16 ):
    """Computes the norm of A by the power method."""
    x = torch.rand( domain_shape, device = 'cuda', generator = torch.Generator(device='cuda').manual_seed(RANDOM_SEED) )
    x_norm = torch.linalg.vector_norm(x)
    x /= x_norm

    norm = torch.sqrt(x_norm)

    for i in range(maxiter):
        x = A_T(A(x))

        x_norm = torch.linalg.vector_norm(x)
        
        norm, norm_old = torch.sqrt(x_norm), norm
        if torch.isclose( norm, norm_old, rtol = rtol, atol = atol ):
            break
        
        x /= x_norm

    return norm

def scalar_multiplication( alpha, op ):
    """Returns an operator rusulting from the scalar multiplication of 'alpha' and 'op'."""
    def result_op(x):
        return alpha * op(x)
    return result_op

def check_ray_transform( T, T_transp, img_shape, sino_shape, tol = 2 * np.finfo(np.float32).eps ):
    """Performs sanity checks on the norm and the transpose operator for the Ray Transform"""

    # Checking the normalization
    T_norm = power_method( T, T_transp, img_shape ).detach().cpu().numpy()
    assert np.isclose( T_norm, 1.0 ), f'T is not normalized ({T_norm} != 1.0)'

    # Checking the transpose operator
    samples = 100
    mean = 0.0
    torch_generator = torch.Generator(device='cuda').manual_seed(RANDOM_SEED)
    for i in range(samples):
        x = torch.rand( img_shape, device = 'cuda', generator = torch_generator )
        y = torch.rand( sino_shape, device = 'cuda', generator = torch_generator )

        Ax = T(x)
        ATy = T_transp(y)

        mean += torch.abs( ( torch.sum( Ax * y ) - torch.sum( x * ATy ) ) / torch.sum( x * ATy ) )
    mean /= samples
    assert mean < tol, f'Transpose operator check failed -> mean error is {mean} > {tol}'

def check_wavelet_transform( W, W_transp, img_shape, tol = 3 * np.finfo(np.float32).eps ):
    """Performs sanity checks on the norm and the transpose operator for the Wavelet Transform"""

    # Numpy random generator
    rng = np.random.default_rng( seed = RANDOM_SEED )

    # Checking orthogonality (numerically)
    samples = 100
    mean = 0.0
    for i in range(samples):
        x = rng.uniform( size = img_shape ).astype(np.float32)
        mean += np.linalg.norm( x - W_transp( W( x ) ) ) / np.linalg.norm(x)
    mean /= samples
    assert mean < tol, f'W is not orthonormal (Numerical orthogonality check failed -> Mean error is {mean} > {tol})'

    # Checking the transpose operator
    samples = 100
    mean = 0.0
    for i in range(samples):
        x = rng.uniform( size = img_shape ).astype(np.float32)
        y = rng.uniform( size = img_shape ).astype(np.float32)

        Wx = W(x)
        WTy = W_transp(y)

        mean += np.abs( ( np.sum( Wx * y ) - np.sum( x * WTy ) ) / np.sum( x * WTy ) )
    mean /= samples
    assert mean < tol, f'Transpose operator check failed -> mean error is {mean} > {tol}'

@tf.custom_gradient
def discrete_gradient_transform( x ):
    """Implements the Discrete Gradient Transform with zero padding"""

    # Differences
    diff_v = x[:,1:,:,:] - x[:,:-1,:,:]
    diff_h = x[:,:,1:,:] - x[:,:,:-1,:]

    # Assumption of zero padding with forward differences
    discr_grad = tf.stack( 
        ( tf.concat( ( diff_v, -x[:,-1:,:,:]), axis = 1 ),
           tf.concat( ( diff_h, -x[:,:,-1:,:]), axis = 2 ) ),
         axis = 1 )
    
    # Explicit formula for the gradient
    def explicit_gradient( du ):
        return discrete_gradient_transform_transp( du )
    
    return discr_grad, explicit_gradient

@tf.custom_gradient
def discrete_gradient_transform_transp( u ):
    """Implements the transpose of the Discrete Gradient Transform with zero padding"""

    # Extracting p and q from u
    p = u[:,0,:,:,:]
    q = u[:,1,:,:,:]

    # Differences
    div_p = p[:,1:,:,:] - p[:,:-1,:,:]
    div_q = q[:,:,1:,:] - q[:,:,:-1,:]

    # Assumption of zero padding with backward differences
    div = tf.concat( ( p[:,0:1,:,:], div_p ), axis = 1 ) + tf.concat( ( q[:,:,0:1,:], div_q ), axis = 2 )
    div *= -1.0

    # Explicit formula for the gradient
    def explicit_gradient( dx ):
        return discrete_gradient_transform( dx )

    return div, explicit_gradient

def check_discrete_gradient_op( img_shape, tol = 1e-5 ):
    """Performs sanity checks on the norm and the transpose operator for the Discrete Gradient Operator"""

    # Checking the transpose operator
    samples = 100
    mean = tf.constant( 0.0, dtype = tf.float32 )
    for i in range(samples):
        x = tf.random.normal( (1,) + img_shape + (1,), dtype = tf.float32, seed = RANDOM_SEED )
        # Checking only for y in the range of the Discrete Gradient Transform
        y = discrete_gradient_transform( tf.random.normal( (1,) + img_shape + (1,), dtype = tf.float32, seed = RANDOM_SEED ) )

        Dx = discrete_gradient_transform(x)
        DTy = discrete_gradient_transform_transp(y)

        mean += tf.math.abs( ( tf.reduce_sum( Dx * y ) - tf.reduce_sum( x * DTy ) ) / tf.reduce_sum( x * DTy ) )
    mean /= samples

    assert mean < tol, f'Transpose operator check failed -> mean error is {mean} > {tol}'

def data_simulation_ray_transform( dataset ):
    """Returns the Ray Transform used to simulate sinograms in each 'dataset'."""

    assert dataset == 'mayo_clinic_128' or dataset == 'mayo_clinic_512'

    # Building the Ray Transform
    if dataset == 'mayo_clinic_128':
        # Avoids inversion crime (original images are 512x512)
        num_detectors = 183
        num_angles = 285
        sino_shape = (num_detectors,num_angles)
        img_shape = ( 512, 512 )
        ( radon, radon_transpose ) = rt.make_radon_transp(
                                        sino_shape        = sino_shape,
                                        sino_top_left     = (      0.0,  sqrt(2.0) * 64 ),
                                        sino_bottom_right = ( torch.pi, -sqrt(2.0) * 64 ),
                                        img_shape         = img_shape,
                                        img_top_left      = (       64,             -64 ),
                                        img_bottom_right  = (      -64,              64 )
                                    )
    else:
        # Does not avoid inversion crime (original images are 512x512)
        num_detectors = 1000
        num_angles = 1000
        sino_shape = (num_detectors,num_angles)
        n = 512
        img_shape = ( n, n )
        ( radon, radon_transpose ) = rt.make_radon_transp(
                                        sino_shape        = sino_shape,
                                        sino_top_left     = ( 0.0, sqrt(2.0) * n / 2 ),
                                        sino_bottom_right = ( torch.pi, -sqrt(2.0) * n / 2 ),
                                        img_shape         = img_shape,
                                        img_top_left      = (   n // 2, - n // 2 ),
                                        img_bottom_right  = ( - n // 2,   n // 2 )
                                    )

    # T norm estimation
    T_norm = power_method( radon, radon_transpose, img_shape )
    # T and T_transp normalization
    T = scalar_multiplication( 1/T_norm, radon)
    T_transp = scalar_multiplication( 1/T_norm, radon_transpose)
    # Sanity check on Ray Transform
    check_ray_transform( T, T_transp, img_shape, sino_shape )

    return T

def make_wavelet_transform( wavelet, level, shape ):
    """Returns the Wavelet Transform from PyWavelets and its transpose"""

    # Getting the coefficient slices for the Wavelet Transform considering the inputs
    _, coeff_slices = pywt.coeffs_to_array( pywt.wavedec2( np.zeros(shape), wavelet, mode = 'periodization', level = level ) )

    # Wavelet Transform and its transpose
    def W( x ):
        return pywt.coeffs_to_array( pywt.wavedec2( x, wavelet, mode = 'periodization', level = level ) )[0]
    def W_transp( c ):
        return pywt.waverec2( pywt.array_to_coeffs( c, coeff_slices, output_format = 'wavedec2' ), wavelet, mode = 'periodization' )
    
    return W, W_transp

def wavelet_transform_coeff_reg( level, shape ):
    """Returns an array with values in the range [0,1] depending on the level of each coefficient.
       The coefficient of the detail/fluctuation at the i-th level has value (level-i+1)/level.
       The coefficient of the trend at the final level has value 0.0 """
    scales = level * np.ones( shape, dtype = np.float32 )
    last_row = shape[0]
    last_column = shape[1]
    for i in range(level):
        last_row //= 2
        last_column //= 2
        scales[ :last_row, :last_column ] -= 1.0
    scales /= level
    return scales

def reconstruction_operators( dataset, problem ):
    """Returns the operators used in the reconstruction, other than shapes, norms, and the regualarization parameter."""

    assert dataset == 'mayo_clinic_128' or dataset == 'mayo_clinic_512'
    assert problem == 'nnls' or problem == 'lasso' or problem == 'slasso' or problem == 'nnslasso' or problem == 'lstv'

    ### The normalized Ray Transform
    if dataset == 'mayo_clinic_128':
        num_detectors = 183
        num_angles = 285
        sino_shape = (num_detectors,num_angles)
        n = 128
        img_shape = ( n, n )
        ( radon, radon_transpose ) = rt.make_radon_transp(
                                        sino_shape        = sino_shape,
                                        sino_top_left     = (      0.0,  sqrt(2.0) * n / 2 ),
                                        sino_bottom_right = ( torch.pi, -sqrt(2.0) * n / 2 ),
                                        img_shape         = img_shape,
                                        img_top_left      = (   n // 2, - n // 2 ),
                                        img_bottom_right  = ( - n // 2,   n // 2 )
                                    )
    else:
        num_detectors = 1000
        num_angles = 1000
        sino_shape = (num_detectors,num_angles)
        n = 512
        img_shape = ( n, n )
        ( radon, radon_transpose ) = rt.make_radon_transp(
                                        sino_shape        = sino_shape,
                                        sino_top_left     = (      0.0,  sqrt(2.0) * n / 2 ),
                                        sino_bottom_right = ( torch.pi, -sqrt(2.0) * n / 2 ),
                                        img_shape         = img_shape,
                                        img_top_left      = (   n // 2, - n // 2 ),
                                        img_bottom_right  = ( - n // 2,   n // 2 )
                                    )
    # T norm estimation
    T_norm = power_method( radon, radon_transpose, img_shape, maxiter = 1000, rtol = 1e-12, atol = 1e-14 )
    # T and T_transp normalization
    T = scalar_multiplication( 1/T_norm, radon)
    T_transp = scalar_multiplication( 1/T_norm, radon_transpose)
    # Sanity check on Ray Transform
    check_ray_transform( T, T_transp, img_shape, sino_shape )
    # Enlarge the norm by 10% as a safeguard for numerical error in the norm estimation
    T_norm = np.float32( 1.1 )
    
    ### Regularization transform
    if problem == 'lasso':
        # Orthonormal Wavelet Transform
        wav = 'sym5'
        if dataset == 'mayo_clinic_128':
            nlevels = int( np.floor( ( pywt.dwtn_max_level( img_shape, wav ) + 1 ) / 2) )
        elif dataset == 'mayo_clinic_512':
            nlevels = 5
        W, W_transp = make_wavelet_transform( wav, nlevels, img_shape )
        # Sanity check on Wavelet Transform
        check_wavelet_transform( W, W_transp, img_shape )
        # Wavelet Transform is orthonormal, thus its norm is equal to 1.0
        W_norm = np.float32( 1.0 )
    elif problem == 'slasso' or problem == 'nnslasso' or problem == 'lstv':
        ## Discrete Gradient Transform
        W = discrete_gradient_transform
        W_transp = discrete_gradient_transform_transp
        # Sanity check on Discrete Gradient Transform
        # check_discrete_gradient_op( img_shape )
        W_norm = np.float32( np.sqrt( 8.0 ) )
    elif problem == 'nnls':
        W = None
        W_norm = np.float32( 0.0 )

    ### Regularization parameter
    if problem == 'lasso':
        lam = np.power( 1.8, wavelet_transform_coeff_reg( nlevels, img_shape ) )
        lam *= np.float32(0.0005 / 2.0)
    elif problem == 'slasso' or problem == 'nnslasso':
        lam = np.float32(0.0015)
    elif problem == 'lstv':
        lam = np.float32(0.0005)
    elif problem == 'nnls':
        lam = np.float32(0.0)
        
    return T, T_transp, img_shape, sino_shape, T_norm, W, W_transp, W_norm, lam


######################## TENSORFLOW - TORCH INTEGRATION ########################
class torch_linear_op_as_tf2_layer(tf.keras.layers.Layer):
    """Returns a Layer built from a torch linear operator: allows auto-differentation."""

    # Note the added `**kwargs`, as Keras supports many arguments
    def __init__(self, torch_op, torch_op_adjoint, domain_shape, range_shape, npdtype, op_name = 'TorchOp', **kwargs):
        super().__init__(**kwargs)
        self.torch_op = torch_op
        self.torch_op_adjoint = torch_op_adjoint
        self.domain_shape = domain_shape
        self.range_shape = range_shape
        self.npdtype = npdtype
        self.op_name = op_name

    @tf.custom_gradient
    def call( self, x ):
        with tf.name_scope(self.op_name):
            # Validate input shape
            x_shape = x.get_shape()
            try:
                # Lazy check if the first dimension is dynamic
                n_x = int(x_shape[0])
                fixed_size = True
            except TypeError:
                n_x = x_shape[0]
                fixed_size = False
            assert x_shape[1:] == self.domain_shape + (1,)
            # Setting input and output shape
            in_shape = (n_x,) + self.domain_shape + (1,)
            out_shape = (n_x,) + self.range_shape + (1,)

            # Some operation that you can only compute with NumPy
            def evaluate_torch_op( x ):
                # Validate input shape
                if fixed_size:
                    x_out_shape = out_shape
                    assert x.shape == in_shape
                else:
                    x_out_shape = (x.shape[0],) + out_shape[1:]
                    assert x.shape[1:] == in_shape[1:]

                # Evaluate the operator on all inputs in the batch.
                out = np.empty(x_out_shape, self.npdtype)

                for i in range(x_out_shape[0]):
                    out_torch = self.torch_op( torch.from_numpy(x[i, ..., 0]).float().cuda() )
                    out[i, ..., 0] = out_torch.detach().cpu().numpy()

                return out
            
            # Calling numpy function
            y = tf.numpy_function(evaluate_torch_op, [x], x.dtype)

            # We must manually set the output shape since tensorflow cannot
            # figure it out
            y.set_shape(out_shape)

            def grad(dy):
                # Validate the input/output shape
                x_shape = x.get_shape()
                dy_shape = dy.get_shape()
                try:
                    # Lazy check if the first dimension is dynamic
                    n_x = int(x_shape[0])
                    fixed_size = True
                except TypeError:
                    n_x = x_shape[0]
                    fixed_size = False

                in_shape = (n_x,) + self.range_shape + (1,)
                out_shape = (n_x,) + self.domain_shape + (1,)

                assert x_shape[1:] == self.domain_shape + (1,)
                assert dy_shape[1:] == self.range_shape + (1,)

                # The gradient of the operation computed with NumPy
                def evaluate_grad_op( x, dy ):
                    # Validate the shape of the given input
                    if fixed_size:
                        x_out_shape = out_shape
                        assert x.shape == out_shape
                        assert dy.shape == in_shape
                    else:
                        x_out_shape = (x.shape[0],) + out_shape[1:]
                        assert x.shape[1:] == out_shape[1:]
                        assert dy.shape[1:] == in_shape[1:]

                    # Evaluate the operator on all inputs in the batch.
                    out = np.empty(x_out_shape, self.npdtype)
                    for i in range(x_out_shape[0]):
                        out_torch = self.torch_op_adjoint( torch.from_numpy(dy[i, ..., 0]).float().cuda() )
                        out[i, ..., 0] = out_torch.detach().cpu().numpy()

                    return out

                # Calling numpy function
                out = tf.numpy_function(evaluate_grad_op, [x, dy], x.dtype)

                # We must manually set the output shape since tensorflow cannot
                # figure it out
                out.set_shape(out_shape)
                return out

            return y, grad
        

######################## TENSORFLOW-NUMPY INTEGRATION ########################
class wavelet_as_tf2_layer(tf.keras.layers.Layer):
    """Returns a Layer build from a numpy linear operator: allows auto-differentation."""

    # Note the added `**kwargs`, as Keras supports many arguments
    def __init__(self, W, W_transp, signal_shape, npdtype, op_name = 'TorchOp', **kwargs):
        super().__init__(**kwargs)
        self.W = W
        self.W_transp = W_transp
        self.signal_shape = signal_shape
        self.npdtype = npdtype
        self.op_name = op_name

    @tf.custom_gradient
    def call( self, x ):
        with tf.name_scope(self.op_name):
            # Validate input shape
            x_shape = x.get_shape()
            try:
                # Lazy check if the first dimension is dynamic
                n_x = int(x_shape[0])
                fixed_size = True
            except TypeError:
                n_x = x_shape[0]
                fixed_size = False
            assert x_shape[1:] == self.signal_shape + (1,)
            # Setting input and output shape
            in_shape = (n_x,) + self.signal_shape + (1,)
            out_shape = (n_x,) + self.signal_shape + (1,)

            # Some operation that you can only compute with NumPy
            def evaluate_wavelet( x ):
                # Validate input shape
                if fixed_size:
                    x_out_shape = out_shape
                    assert x.shape == in_shape
                else:
                    x_out_shape = (x.shape[0],) + out_shape[1:]
                    assert x.shape[1:] == in_shape[1:]

                # Evaluate the operator on all inputs in the batch.
                out = np.empty(x_out_shape, self.npdtype)

                for i in range(x_out_shape[0]):
                    out[i, ..., 0] = self.W( x[i, ..., 0] )

                return out
            
            # Calling numpy function
            y = tf.numpy_function(evaluate_wavelet, [x], x.dtype)

            # We must manually set the output shape since tensorflow cannot
            # figure it out
            y.set_shape(out_shape)

            def grad(dy):
                # Validate the input/output shape
                x_shape = x.get_shape()
                dy_shape = dy.get_shape()
                try:
                    # Lazy check if the first dimension is dynamic
                    n_x = int(x_shape[0])
                    fixed_size = True
                except TypeError:
                    n_x = x_shape[0]
                    fixed_size = False

                in_shape = (n_x,) + self.signal_shape + (1,)
                out_shape = (n_x,) + self.signal_shape + (1,)

                assert x_shape[1:] == self.signal_shape + (1,)
                assert dy_shape[1:] == self.signal_shape + (1,)

                # The gradient of the operation computed with NumPy
                def evaluate_grad_op( x, dy ):
                    # Validate the shape of the given input
                    if fixed_size:
                        x_out_shape = out_shape
                        assert x.shape == out_shape
                        assert dy.shape == in_shape
                    else:
                        x_out_shape = (x.shape[0],) + out_shape[1:]
                        assert x.shape[1:] == out_shape[1:]
                        assert dy.shape[1:] == in_shape[1:]

                    # Evaluate the operator on all inputs in the batch.
                    out = np.empty(x_out_shape, self.npdtype)
                    for i in range(x_out_shape[0]):
                        out[i, ..., 0] = self.W_transp( dy[i, ..., 0] )

                    return out

                # Calling numpy function
                out = tf.numpy_function(evaluate_grad_op, [x, dy], x.dtype)

                # We must manually set the output shape since tensorflow cannot
                # figure it out
                out.set_shape(out_shape)
                return out

            return y, grad