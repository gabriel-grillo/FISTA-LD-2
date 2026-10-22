#include "cuda_radon.h"
#include "cuda_image2.h"

#define SYNC_AND_CHECK \
cudaDeviceSynchronize(); \
if ( cudaGetLastError() != cudaSuccess ) \
{ \
   printf( "Kernel launch failed above line %d: %s\n", __LINE__, cudaGetErrorString( cudaGetLastError() ) ); \
   std::exit( EXIT_FAILURE ); \
}

typedef cuda::image2_t< float, int > image_t;
dim3 blck_dim_radon = dim3( 32, 20 );
dim3 grid_dim_radon = dim3( 19, 19 );
dim3 blck_dim_trans = dim3( 32, 20 );
dim3 grid_dim_trans = dim3( 19, 19 );

extern "C" {
   void radon( image_t image, image_t sino );
   void radon_transpose( image_t sino, image_t image );
} //extern "C"

void radon( image_t image, image_t sino )
{
   // Compute Radon transform on device:
   cuda::radon< 110 ><<< grid_dim_radon, blck_dim_radon >>>( image, sino );
   SYNC_AND_CHECK
}

void radon_transpose( image_t sino, image_t image )
{
   // Compute transpose Radon transform on device:
   cuda::radon_transpose< 110 ><<< grid_dim_trans, blck_dim_trans >>>( sino, image );
   SYNC_AND_CHECK
}
