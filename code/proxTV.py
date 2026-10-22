import tensorflow as tf
from math import inf

def reduce_TV( u ):
    return tf.reduce_sum( tf.math.abs( u ), axis = [1,2,3,4] )

def dual_projection( u ):
    return u / tf.math.maximum( tf.math.abs(u), 1.0 )

class proxTV():
    def __init__(self, D, D_transp, D_norm, lam, lower = -inf, upper = inf, MAXITER = 10):
        self.D = D
        self.D_transp = D_transp
        self.D_norm_sq = D_norm ** 2
        self.lam = lam
        self.lower = lower
        self.upper = upper
        self.MAXITER = MAXITER

    # Projecting the image onto the box
    def box_projection( self, x ):
        return tf.math.minimum( tf.math.maximum( x, self.lower ), self.upper )

    def __call__(self, z, rho, eps):
        # Initialization of the dual variables (shared across outer iterations)
        if not hasattr( self, 'u' ):
            batch_size = tf.shape(z)[0]
            img_shape = tf.shape(z)[1:]
            dual_shape = tf.concat( [[ batch_size, 2 ], img_shape], axis = 0 )
            self.u = tf.zeros( dual_shape, dtype = tf.float32, name = 'u' )
            self.dual_gap = tf.zeros( (batch_size,), dtype = tf.float32, name = 'dualgap' ) #just to set the shape

        # Initializations
        i = tf.constant( 0, dtype = tf.int32, name = 'proxTV_i' )
        t = tf.constant( 1, dtype = tf.float32, name = 'proxTV_t' )
        x = tf.zeros_like( z, name = 'proxTV_x' ) #just to set the shape
        w = self.u

        # Dual stepsize
        beta = 1 / ( rho * self.D_norm_sq )

        # Flag to indicate whether the stopping criterion have been met (for each batch item)
        flag = tf.zeros( tf.shape(self.dual_gap), dtype = tf.bool, name = 'flag' )

        # Iteration counter for each batch item
        iter_batch = tf.zeros( tf.shape(self.dual_gap), dtype = tf.int32, name = 'iter_batch' )

        while tf.logical_and( tf.less(i, self.MAXITER), tf.logical_not( tf.math.reduce_all( flag ) ) ):
            # Incrementing iteration counter
            i += 1
            iter_batch += tf.where( flag, 0, 1 )

            # Mask for the indices to update: only where the stopping criterion have not yet been met
            mask = tf.where( flag[:,tf.newaxis,tf.newaxis,tf.newaxis,tf.newaxis], tf.zeros_like( self.u ), tf.ones_like( self.u ) )

            # Previous iteration information
            u_prev = self.u
            
            # Compute gradient of dual objective function
            grad = self.D( self.box_projection( z - rho * self.D_transp( w ) ) )
        
            # Update current iteration towards gradient direction
            self.u = w + beta * mask * grad
        
            # Project into feasible dual set (proximal operator of the indicator function)
            self.u = self.lam * dual_projection( self.u / self.lam )
            
            # t for next iteration
            t_next = (1.0 + tf.sqrt(1.0 + 4.0 * t**2)) / 2.0
            
            # Momentum update
            w = self.u + ((t-1)/t_next) * ( self.u - u_prev )
            
            # t update
            t = t_next

            # Current primal approximation
            D_transp_u = self.D_transp( self.u )
            x = D_transp_u
            x *= -rho
            x += z
            x = self.box_projection( x )

            # Updating duality gap
            self.dual_gap = self.lam * reduce_TV( self.D( x ) ) - tf.reduce_sum( x * D_transp_u, axis = [1,2,3] )

            # Stopping criterion flag for each batch item
            flag = tf.math.less( self.dual_gap, eps )

        return x, tf.where( flag, 1, 0 ), iter_batch
