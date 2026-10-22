# I use the one encoding: utf8
import os
import ctypes
import ctypes.util
import numpy
import math
import pyraft as pr
import torch

# Load required libraries:
libcudart = ctypes.CDLL( ctypes.util.find_library( "cudart" ), mode=ctypes.RTLD_GLOBAL )
libstdcpp = ctypes.CDLL( ctypes.util.find_library( "stdc++" ), mode=ctypes.RTLD_GLOBAL )
libtomo   = ctypes.CDLL( os.path.join( os.path.dirname(__file__), "cuda_radon_torch.so" ) )

# "float *" type:
_c_float_p = ctypes.POINTER( ctypes.c_float )

class RAFT_MATRIX( ctypes.Structure ):
   """A raft_matrix from raft:"""
   _fields_ = [ ( "p_data_device", _c_float_p ),
                ( "p_data_host", _c_float_p ),
                ( "lines", ctypes.c_int ),
                ( "columns", ctypes.c_int )
             ]

class RAFT_IMAGE( ctypes.Structure ):
   """A raft_image from raft:"""
   _fields_ = [ ( "data", RAFT_MATRIX ),
                ( "tl_x", ctypes.c_float ),
                ( "tl_y", ctypes.c_float ),
                ( "br_x", ctypes.c_float ),
                ( "br_y", ctypes.c_float )
              ]

def make_RAFT_MATRIX( array ):
   """Make a raft_matrix from a torch.Tensor"""
   return RAFT_MATRIX(
      ctypes.cast( array.data_ptr(), _c_float_p ),
      ctypes.cast( 0, _c_float_p ),
      ctypes.c_int( array.shape[ 0 ] ),
      ctypes.c_int( array.shape[ 1 ] )
   )

def make_RAFT_IMAGE( array, top_left = None, bottom_right = None ):
   """Make a raft_image from a torch.Tensor"""

   if top_left is None:
      top_left = ( -1.0, 1.0 )
   if bottom_right is None:
      bottom_right = ( 1.0, -1.0 )

   return RAFT_IMAGE(
      make_RAFT_MATRIX( array ),
      ctypes.c_float( top_left[ 0 ] ),
      ctypes.c_float( top_left[ 1 ] ),
      ctypes.c_float( bottom_right[ 0 ] ),
      ctypes.c_float( bottom_right[ 1 ] )
   )

# Function prototypes:
libtomo.radon.argtypes = [ RAFT_IMAGE, RAFT_IMAGE ]
libtomo.radon_transpose.argtypes = [ RAFT_IMAGE, RAFT_IMAGE ]

def make_radon_transp(
   sino_shape,
   sino_top_left = ( 0.0, 1.0 ), sino_bottom_right = ( math.pi, -1.0 ),
   img_shape = None,
   img_top_left = None, img_bottom_right = None
   ):

   if img_shape is None:
      img_shape = ( sino_shape[ 0 ], sino_shape[ 0 ] )

   def radon( x ):
      """
         Compute projection through ray-tracing techniques
      """

      # Create room for output
      sino_data = torch.zeros( sino_shape, device = 'cuda' )
      torch.cuda.synchronize()

      # Create output argument for C function
      SINO = make_RAFT_IMAGE( sino_data, sino_top_left, sino_bottom_right )

      # Create input argument for C function
      IMAGE = make_RAFT_IMAGE( x, img_top_left, img_bottom_right )

      # Call C function
      libtomo.radon( IMAGE, SINO )

      # Return result
      return sino_data

   def radon_transpose( y ):
      """
         Compute backprojection through ray-tracing techniques
      """

      image_data = torch.zeros( img_shape, device = 'cuda' )
      torch.cuda.synchronize()
      IMAGE = make_RAFT_IMAGE( image_data, img_top_left, img_bottom_right )

      SINO = make_RAFT_IMAGE( y, sino_top_left, sino_bottom_right )

      libtomo.radon_transpose( SINO, IMAGE )

      return image_data

   return radon, radon_transpose
