import tensorflow as tf
from proxTV import proxTV, reduce_TV
from math import inf

#############################
# Optimize F(x) = f(x) + g(x)
#############################

##### Helpers

def dot(x,y):
    dot = tf.reduce_sum( x * y, axis=[1,2,3] )
    return dot
def norm_sq(x):
    norm = tf.reduce_sum( x**2, axis = [1,2,3] )
    return norm
def safe_sqrt(x):
    safe_sqrt = tf.sqrt(tf.where(tf.equal(x, 0.0), x + 1, x))
    return tf.where(tf.equal(x, 0.0), tf.zeros_like(x), safe_sqrt)
## Safe normalization ensures that there is no division by 0 in backpropagation
def safely_normalize( x, M ):
    sqnorm = tf.reduce_sum( x**2, axis=[1,2,3], keepdims=True )
    safe_sqrt = tf.sqrt(sqnorm + M ** 2)
    x = M * x / safe_sqrt
    return x

##### Objective function terms
def huber(inp, gamma):
    batch_size = tf.shape(inp)[0]
    y = tf.where(tf.abs(inp) < gamma, 
                 0.5 * inp**2 / gamma, 
                 tf.abs(inp) - 0.5 * gamma)
    return tf.reduce_sum( tf.reshape( y, shape = [batch_size, -1] ), axis = -1 )
def huber_grad(inp, gamma):
    return tf.where(tf.abs(inp) < gamma, inp / gamma, tf.sign(inp))

def l1( inp ):
    out = tf.reduce_sum( tf.abs( inp ), axis = [1,2,3] )
    return out
def prox_l1(x, alpha):
    res = tf.sign(x) * tf.maximum(tf.abs(x) - alpha, 0)
    return res

def objective_function( problem, A, A_adj, A_norm, b, W, W_adj, W_norm, lam ):
    if problem == 'nnls':
        # Nonnegative least squares problem
        def f( x ):
            r = A( x ) - b
            def grad():
                return A_adj( r )
            return 0.5*norm_sq( r ), grad
        L = A_norm**2
        def g( x ):
            return 0.0 # I always project anyway
        def prox( inp, step_size, eps ):
            out = tf.math.maximum( 0.0, inp )
            return out, tf.fill( [tf.shape(inp)[0]], 1 ), tf.fill( [tf.shape(inp)[0]], 1 )
        def proj_C( inp ):
            out = tf.math.maximum( 0.0, inp )
            return out
    elif problem == 'lasso':
        # Lasso problem: W-l1 regularized least squares problem
        def f( x ):
            r = A( x ) - b
            def grad():
                return A_adj( r )
            return 0.5*norm_sq( r ), grad
        L = A_norm**2
        def g( x ):
            return l1( lam * W(x) )
        def prox(x, rho, eps):
            return W_adj( prox_l1( W(x), lam * rho ) ), tf.fill( [tf.shape(x)[0]], 1 ), tf.fill( [tf.shape(x)[0]], 1 )
        def proj_C( inp ):
            return inp
    elif problem == 'slasso':
        # Smoothed Lasso problem: W-l1 (smoothed) regularized least squares problem
        huber_param = tf.constant( 1e-2, dtype = tf.float32, name = 'huber_param' )
        def f( x ):
            r = A( x ) - b
            Wx = W( x )
            def grad():
                return A_adj( r ) + lam * W_adj( huber_grad( Wx, huber_param ) )
            return 0.5 * norm_sq( r ) + lam * huber( Wx, huber_param ), grad
        L = A_norm**2 + lam * (W_norm**2) / huber_param
        def g( x ):
            return 0.0
        def prox( x, rho, eps):
            return x, tf.fill( [tf.shape(x)[0]], 1 ), tf.fill( [tf.shape(x)[0]], 1 )
        def proj_C( inp ):
            return inp
    elif problem == 'nnslasso':
        # Nonnegative smoothed Lasso problem: W-l1 (smoothed) regularized least squares problem
        huber_param = tf.constant( 1e-2, dtype = tf.float32, name = 'huber_param' )
        def f( x ):
            r = A( x ) - b
            Wx = W(x)
            def grad():
                return A_adj( r ) + lam * W_adj( huber_grad( Wx, huber_param ) )
            return 0.5*norm_sq( r ) + lam * huber( Wx, huber_param ), grad
        L = A_norm**2 + lam * (W_norm**2) / huber_param
        def g( x ):
            return 0.0 # I always project anyway
        def prox( inp, step_size, eps ):
            out = tf.math.maximum( 0.0, inp )
            return out, tf.fill( [tf.shape(inp)[0]], 1 ), tf.fill( [tf.shape(inp)[0]], 1 )
        def proj_C( inp ):
            out = tf.math.maximum( 0.0, inp )
            return out
    elif problem == 'lstv':
        lower = -inf
        upper = + inf
        def f( x ):
            r = A( x ) - b
            def grad():
                return A_adj( r )
            return 0.5*norm_sq( r ), grad
        L = A_norm**2
        def g( x ):
            return lam * reduce_TV( W( x ) )
        prox = proxTV( W, W_adj, W_norm, lam, lower = lower, upper = upper, MAXITER = 10 )
        def proj_C( x ):
            if lower > -inf:
                x = tf.math.maximum( x, lower )
            if upper < inf:
                x = tf.math.minimum( x, upper )
            return x
    else:
        raise ValueError('Unimplemented problem')

    return f, g, prox, proj_C, L


################# Computing R0 for fista_ld
def computeR( problem, x0, A, A_adj, A_norm, b, W, W_adj, W_norm, lam ):
    # Objective funtion terms
    f, _, prox, _, L = objective_function( problem, A, A_adj, A_norm, b, W, W_adj, W_norm, lam )
    # Performing one proximal gradient step with loose proximal tolerance
    _, gradx = f( x0 )
    grad = gradx()
    x1, _, _ = prox( x0 - (1/L) * grad, 1/L, tf.constant( 1e+7, dtype = tf.float32 ) )
    # Computing squared norm of the proximal gradient step
    R = tf.sqrt( L * norm_sq( x1 - x0 ) )
    return R


########################## FISTA with Learned Deviations ##########################
def fista_ld( problem, x, A, A_adj, A_norm, b, W, W_adj, W_norm, lam,
              d_model, e_model, v_model, theta,
              MAXITER, MAXITER_DEVS, gamma, alpha, mu = 0.99,
              training = False, save_hist = True
             ):
    
    # axis=0(batch), axes=1,2(2d image), axis=3(channels)
    batch_size = tf.shape(x)[0]
    
    ## Objective funtion terms
    f, g, prox, proj_C, L = objective_function( problem, A, A_adj, A_norm, b, W, W_adj, W_norm, lam )

    ## Stepsize limits
    c_min = 0.1
    c_max = 10.0
    theta_center = tf.math.log( ( c_max - 1 ) / ( 1 - c_min ) )
    rho_theory = mu / L
    rho_min = c_min * rho_theory
    rho_max = c_max * rho_theory
    
    ## Initializations
    i = tf.constant(0, dtype=tf.int32, name = 'i')
    t = tf.constant(1.0, dtype=tf.float32, name = 't')
    z = x
    R0 = computeR( problem, x, A, A_adj, A_norm, b, W, W_adj, W_norm, lam )
    d = tf.zeros_like( x )
    e = tf.zeros_like( x )
    v = tf.zeros_like( x )
    flag = tf.fill( [batch_size], 0, name = 'flag' )
    
    if save_hist:
        Fhist           = tf.fill( [ batch_size, MAXITER + 1 ], 0.0, name = 'Fhist'           )
        eps_hist        = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'eps_hist'        )
        d_hist          = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'd_hist'          )
        d_ub_hist       = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'd_ub_hist'       )
        e_hist          = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'e_hist'          )
        e_ub_hist       = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'e_ub_hist'       )
        v_hist          = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'v_hist'          )
        v_ub_hist       = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'v_ub_hist'       )
        flag_hist       = tf.fill( [ batch_size, MAXITER     ],   0, name = 'flag_hist'       )
        inner_iter_hist = tf.fill( [ batch_size, MAXITER     ],   0, name = 'inner_iter_hist' )

        fx, _ = f( x )
        Fx = fx + g( x )
        Fhist += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( Fx, axis = 1 )
    else:
        Fhist           = []
        eps_hist        = []
        d_hist          = []
        d_ub_hist       = []
        e_hist          = []
        e_ub_hist       = []
        v_hist          = []
        v_ub_hist       = []
        flag_hist       = []
        inner_iter_hist = []

    while tf.less(i, MAXITER):
        ## i update
        i += 1

        ## Saves previous iteration
        x_prev = x

        ## Current rho
        if tf.less_equal( i, MAXITER_DEVS ):
            current_rho = rho_min + (rho_max - rho_min) * tf.math.sigmoid( theta[i-1] - theta_center )
        else:
            current_rho = rho_theory

        ## y update
        y = ( 1 - 1 / t ) * x + ( 1 / t ) * z

        ## x update

        # Current gradient
        _, grad_y = f( y )
        grad = grad_y()

        # d deviation
        d_norm_upper_bound = R0 / tf.cast( i, dtype = tf.float32 ) ** gamma
        d_norm_upper_bound /= t
        if tf.less_equal(i, MAXITER_DEVS):
            d = d_model( [ y, grad, d ], training = training )
            d = safely_normalize( d, d_norm_upper_bound[:,tf.newaxis,tf.newaxis,tf.newaxis] )
        else:
            d = tf.zeros_like( x )

        # Gradient step with deviation
        grad += d
        grad *= current_rho
        y -= grad

        # Proximal step
        eps = R0 / tf.cast( i, dtype = tf.float32 ) ** gamma
        eps /= t ** 2
        x, flag, inner_iter = prox( y, current_rho, eps )

        # e deviation
        e_norm_upper_bound = R0 / tf.cast( i, dtype = tf.float32 ) ** gamma
        e_norm_upper_bound /= t
        if tf.less_equal(i, MAXITER_DEVS):
            e = e_model( [ y, ( y - x ) / current_rho, d, e ], training = training )
            e = safely_normalize( e, e_norm_upper_bound[:,tf.newaxis,tf.newaxis,tf.newaxis] )
        else:
            e = tf.zeros_like(x)
        # Adding e deviation
        x += e
        x = proj_C( x )

        ## z update
        if tf.less( i, MAXITER ):
            # v deviation
            v_norm_upper_bound = R0 / tf.cast( i, dtype = tf.float32 ) ** gamma
            if tf.less(i, MAXITER_DEVS):
                v = v_model( [ x, x - x_prev, d, e, v ], training = training )
                v = safely_normalize( v, v_norm_upper_bound[:,tf.newaxis,tf.newaxis,tf.newaxis] )
            else:
                v = tf.zeros_like(x)
            # Inertial step with deviation
            z = proj_C( x + ( t - 1 ) * ( x - x_prev ) + v )
        else:
            v = tf.zeros_like(x)
            v_norm_upper_bound = 0.0 * R0

        ## t update
        if tf.less( i, MAXITER_DEVS ):
            next_rho = rho_min + (rho_max - rho_min) * tf.math.sigmoid( theta[i] - theta_center )
        else:
            next_rho = rho_theory
        t = tf.math.maximum( 1.0, ( 1 - alpha ) / 2 + tf.sqrt( current_rho / next_rho ) * t )
        
        ## Saves history
        if save_hist:
            fx,_ = f(x)
            Fx = fx + g(x)
            Fhist     += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( Fx, axis = 1 )
            eps_hist  += tf.one_hot( i - 1, MAXITER ) * tf.expand_dims( eps, axis = 1 )
            d_hist    += tf.one_hot( i - 1, MAXITER ) * tf.expand_dims( tf.sqrt( tf.reduce_sum( d**2, axis = [1,2,3] ) ), axis = 1 )
            d_ub_hist += tf.one_hot( i - 1, MAXITER ) * tf.expand_dims( d_norm_upper_bound, axis = 1 )
            e_hist    += tf.one_hot( i - 1, MAXITER ) * tf.expand_dims( tf.sqrt( tf.reduce_sum( e**2, axis = [1,2,3] ) ), axis = 1 )
            e_ub_hist += tf.one_hot( i - 1, MAXITER ) * tf.expand_dims( e_norm_upper_bound, axis = 1 )
            v_hist    += tf.one_hot( i - 1, MAXITER ) * tf.expand_dims( tf.sqrt( tf.reduce_sum( v**2, axis = [1,2,3] ) ), axis = 1 )
            v_ub_hist += tf.one_hot( i - 1, MAXITER ) * tf.expand_dims( v_norm_upper_bound, axis = 1 )
            flag_hist += tf.one_hot( i - 1, MAXITER, dtype = tf.int32 ) * tf.expand_dims( flag, axis = 1 )
            inner_iter_hist += tf.one_hot( i - 1, MAXITER, dtype = tf.int32 ) * tf.expand_dims( inner_iter, axis = 1 )

    # Final functional value
    fx,_ = f(x)
    Fx = fx + g(x)
    
    return x, Fx, [Fhist, eps_hist, d_hist, d_ub_hist, e_hist, e_ub_hist, v_hist, v_ub_hist, flag_hist, inner_iter_hist]

    
########################## Baseline algorithms ##########################
def fista( problem, x, A, A_adj, A_norm, b, W, W_adj, W_norm, lam,
           MAXITER, gamma, alpha, mu = 0.99,
           save_hist = True
          ):
    
    # axis=0(batch), axes=1,2(2d image), axis=3(channels)
    batch_size = tf.shape(x)[0]
    
    ## Objective funtion terms
    f, g, prox, proj_C, L = objective_function( problem, A, A_adj, A_norm, b, W, W_adj, W_norm, lam )
    
    ## Initializations
    i = tf.constant(0, dtype=tf.int32, name = 'i')
    t = tf.constant(1.0, dtype=tf.float32, name = 't')
    z = x
    rho = mu / L
    if problem == 'lstv':
        R0 = computeR( problem, x, A, A_adj, A_norm, b, W, W_adj, W_norm, lam )
    else:
        R0 = tf.fill( [ batch_size ], 0.0, name = 'eps_hist' )
    flag = tf.fill( [batch_size], 0, name = 'flag' )
    
    if save_hist:
        Fhist           = tf.fill( [ batch_size, MAXITER + 1 ], 0.0, name = 'Fhist'           )
        eps_hist        = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'eps_hist'        )
        flag_hist       = tf.fill( [ batch_size, MAXITER     ],   0, name = 'flag_hist'       )
        inner_iter_hist = tf.fill( [ batch_size, MAXITER     ],   0, name = 'inner_iter_hist' )

        fx, _ = f( x )
        Fx = fx + g( x )
        Fhist += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( Fx, axis = 1 )
    else:
        Fhist           = []
        flag_hist       = []
        inner_iter_hist = []

    while tf.less(i, MAXITER):
        ## Saves previous iteration
        x_prev = x

        ## i update
        i += 1

        ## y update
        y = ( 1 - 1 / t ) * x + ( 1 / t ) * z

        ## x update

        # Current gradient
        _, grad_y = f( y )
        grad = grad_y()

        # Gradient step with deviation
        w = y - rho * grad

        # Proximal step
        eps = R0 / tf.cast( i, dtype = tf.float32 ) ** gamma
        eps /= t ** 2
        x, flag, inner_iter = prox( w, rho, eps )

        ## z update
        # Inertial step
        z = proj_C( x + ( t - 1 ) * ( x - x_prev ) )

        ## t update (constant step-size)
        t += (1-alpha)/2
        
        ## Saves history
        if save_hist:
            fx,_ = f(x)
            Fx = fx + g(x)
            Fhist     += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( Fx, axis = 1 )
            eps_hist  += tf.one_hot( i - 1, MAXITER ) * tf.expand_dims( eps, axis = 1 )
            flag_hist += tf.one_hot( i - 1, MAXITER, dtype = tf.int32 ) * tf.expand_dims( flag, axis = 1 )
            inner_iter_hist += tf.one_hot( i - 1, MAXITER, dtype = tf.int32 ) * tf.expand_dims( inner_iter, axis = 1 )

    # Final functional value
    fx,_ = f(x)
    Fx = fx + g(x)
    
    return x, Fx, [Fhist, eps_hist, flag_hist, inner_iter_hist]

def ista( problem, x, A, A_adj, A_norm, b, W, W_adj, W_norm, lam,
          MAXITER, save_hist = True
         ):
    # axis=0(batch), axes=1,2(2d image), axis=3(channels)
    batch_size = tf.shape(x)[0]
    
    # Objective funtion terms
    f, g, prox, _, L = objective_function( problem, A, A_adj, A_norm, b, W, W_adj, W_norm, lam )

    # Initializations
    i = tf.constant(0, dtype=tf.int32, name = 'i')
    if save_hist:
        Fhist           = tf.fill( [ batch_size, MAXITER + 1 ], 0.0, name = 'Fhist'           )
        eps_hist        = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'eps_hist'           )
        flag_hist       = tf.fill( [ batch_size, MAXITER     ],   0, name = 'flag_hist'       )
        inner_iter_hist = tf.fill( [ batch_size, MAXITER     ],   0, name = 'inner_iter_hist' )
    else:
        Fhist           = []
        eps_hist        = []
        flag_hist       = []
        inner_iter_hist = []

    while tf.less(i, MAXITER):
        ## Performs proximal gradient step

        # Current gradient
        fx, gradx = f( x )
        grad = gradx()

        # Saves Fx
        if save_hist:
            Fx = fx + g(x)
            Fhist += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( Fx, axis = 1 )

        # Proximal gradient step
        x, flag, inner_iter = prox( x - (1/L) * grad, 1/L, tf.constant( 0.0, dtype = tf.float32 ) )

        ## Updates i for next iteration
        i += 1

        ## Saves hist
        if save_hist:
            flag_hist += tf.one_hot( i - 1, MAXITER, dtype = tf.int32 ) * tf.expand_dims( flag, axis = 1 )
            inner_iter_hist += tf.one_hot( i - 1, MAXITER, dtype = tf.int32 ) * tf.expand_dims( inner_iter, axis = 1 )

    # Final functional value
    fx,_ = f(x)
    Fx = fx + g(x)

    # Saves Fx
    if save_hist:
        Fhist += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( Fx, axis = 1 )
    
    return x, Fx, [Fhist, eps_hist, flag_hist, inner_iter_hist]


########################## Deep Optimization ##########################
##### based on implementations at https://github.com/JevgenijaAksjonova/Deep-Optimization
def deepopt_nonsmooth( problem, x, A, A_adj, A_norm, b, W, W_adj, W_norm, lam,
                       dx1_model, dx2_model, theta, MAXITER, MAXITER_DEVS, alpha,
                       training = False, save_hist = True ):
    # axis=0(batch), axes=1,2(2d image), axis=3(channels)
    batch_size = tf.shape(x)[0]
    
    # Objective funtion terms
    f, g, prox, _, L = objective_function( problem, A, A_adj, A_norm, b, W, W_adj, W_norm, lam )

    # Initializations
    i = tf.constant(0, dtype=tf.int32, name = 'i')
    x_prev = x
    fx, grad_w = f(x)
    grad = grad_w()
    dx1 = tf.zeros_like(x)
    dx2 = tf.zeros_like(x)
    beta = 1/L    # Inverse of the smoothness parameter
    gamma = beta  # step-size
    omega = alpha # alpha_2

    ## Stepsize limits
    c_min = 0.1
    c_max = 2.0
    theta_center = tf.math.log( ( c_max - 1 ) / ( 1 - c_min ) )
    gamma_min = c_min * beta
    gamma_max = c_max * beta # gamma in (0,2beta)

    if save_hist:
        Fhist           = tf.fill( [ batch_size, MAXITER + 1 ], 0.0, name = 'Fhist'           )
        dx1_hist        = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'dx1_hist'        )
        dx1_ub_hist     = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'dx1_ub_hist'     )
        dx2_hist        = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'dx2_hist'        )
        dx2_ub_hist     = tf.fill( [ batch_size, MAXITER     ], 0.0, name = 'dx2_ub_hist'     )
        flag_hist       = tf.fill( [ batch_size, MAXITER     ],   0, name = 'flag_hist'       )
        inner_iter_hist = tf.fill( [ batch_size, MAXITER     ],   0, name = 'inner_iter_hist' )

        Fx = fx + g(x)
        Fhist += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( Fx, axis = 1 )
    else:
        Fhist           = []
        dx1_hist        = []
        dx1_ub_hist     = []
        dx2_hist        = []
        dx2_ub_hist     = []
        flag_hist       = []
        inner_iter_hist = []

    while tf.less(i, MAXITER):
        ## Step-size
        if tf.less(i, MAXITER_DEVS):
            gamma = gamma_min + (gamma_max - gamma_min) * tf.math.sigmoid( theta[i] - theta_center )
        else:
            gamma = beta
        ## Deviations
        dx1_prev = dx1
        dx2_prev = dx2
        # Compute dx1 and w
        c1 = tf.sqrt(2 * beta * alpha * (2 * beta - gamma) / (2 * beta * gamma))
        c1 *=  safe_sqrt(norm_sq( x - x_prev - (beta / (2 * beta - gamma)) * dx2_prev ))
        if tf.less( i, MAXITER_DEVS ):
            dx1 = dx1_model( [x, grad, dx1], training = training )
            dx1 = safely_normalize( dx1, c1[:,tf.newaxis,tf.newaxis,tf.newaxis]  )
        else:
            dx1 = tf.zeros_like( x )
        w = x + dx1
        # Compute current gradient
        grad_prev = grad
        _, grad_w = f(w)
        grad = grad_w()
        # Compute dx2
        c2 = tf.sqrt(2 * gamma * (2 * beta - gamma) / beta * beta * omega / 2)
        c2 *= safe_sqrt(norm_sq( grad - grad_prev - (1 / beta) * (x - x_prev - dx1_prev) ))
        if tf.less( i, MAXITER_DEVS ):
            dx2 = dx2_model( [x, grad, dx2, dx1], training = training )
            dx2 = safely_normalize( dx2, c2[:,tf.newaxis,tf.newaxis,tf.newaxis] )
        else:
            dx2 = tf.zeros_like( x )
        ## Performs proximal gradient step
        x_prev = x
        x -= gamma * grad
        x += (gamma / beta) * dx1
        x += dx2
        x, flag, inner_iter = prox( x, gamma, tf.constant( 0.0, dtype = tf.float32 ) )

        ## Updates i for next iteration
        i += 1

        # Saves Fx
        if save_hist:
            fx,_ = f(x)
            Fx = fx + g(x)
            Fhist       += tf.one_hot( i, MAXITER + 1 ) * tf.expand_dims( Fx, axis = 1 )
            dx1_hist    += tf.one_hot( i-1, MAXITER ) * tf.expand_dims( tf.sqrt( tf.reduce_sum( dx1**2, axis = [1,2,3] ) ), axis = 1 )
            dx1_ub_hist += tf.one_hot( i-1, MAXITER ) * tf.expand_dims( c1, axis = 1 )
            dx2_hist    += tf.one_hot( i-1, MAXITER ) * tf.expand_dims( tf.sqrt( tf.reduce_sum( dx2**2, axis = [1,2,3] ) ), axis = 1 )
            dx2_ub_hist += tf.one_hot( i-1, MAXITER ) * tf.expand_dims( c2, axis = 1 )
            flag_hist += tf.one_hot( i - 1, MAXITER, dtype = tf.int32 ) * tf.expand_dims( flag, axis = 1 )
            inner_iter_hist += tf.one_hot( i - 1, MAXITER, dtype = tf.int32 ) * tf.expand_dims( inner_iter, axis = 1 )
    
    # Final functional value
    fx,_ = f(x)
    Fx = fx + g(x)
    
    return x, Fx, [Fhist, dx1_hist, dx1_ub_hist, dx2_hist, dx2_ub_hist, flag_hist, inner_iter_hist ]
