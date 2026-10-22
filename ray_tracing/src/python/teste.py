import ray_tracing_cuda_float_torch as rt
import torch

import matplotlib.pyplot as pp

n = 2048
img = torch.zeros( ( n, n ), device = 'cuda' )
img[ int( n / 4 ) : int( n / 1.5 ), int( n / 3 ) : int( n / 2 ) ] += 1.0
img[ int( n / 2.5 ) : int( n / 1.25 ), int( n / 9 ) : int( n / 1.2 ) ] += 1.0

( radon, radon_transpose ) = rt.make_radon_transp(
    ( n, n ),
    sino_top_left = ( 0.0, 1.0 ),
    sino_bottom_right = ( torch.pi, -1.0 ),
    img_shape = ( n, n )    
)

for i in range( 100 ):
    b = radon( img )
