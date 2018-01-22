############################################################################
# <summary> CMakeLists.txt for implementations of fundamental GPU algorithms
#           from Udacity lectures. </summary>
# <date>    2018-01-21          </date>
# <author>  Tanmay Bangalore  </author>
############################################################################

# Collect files
file( GLOB  hdr *.hpp *.h )
file( GLOB  cu  *.cu )
file( GLOB  src *.cpp )
    
CUDA_ADD_EXECUTABLE(GPUAlgos ${src} ${hdr} ${cu})