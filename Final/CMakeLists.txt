# Build executables for the CS344 Final

#########
# Smooth
#########

file(GLOB hdr smooth/*.h smooth/*.hpp)
file(GLOB src smooth/*.cu)

CUDA_ADD_EXECUTABLE(smooth ${src} ${hdr})