cmake_minimum_required(VERSION 3.9)

#----------------------------------------------------------------------------
#   Define the project name and declare the use of C++ and CUDA.
#----------------------------------------------------------------------------
project(vectorsSum LANGUAGES CXX CUDA)

#----------------------------------------------------------------------------
#   Define the static library vectors.
#----------------------------------------------------------------------------
add_library(vectors STATIC
    vectorsum.cu
    vectorsum.h
)

#----------------------------------------------------------------------------
#   Declare to use C++11 std.
#----------------------------------------------------------------------------
target_compile_features(vectors PUBLIC cxx_std_11)

#----------------------------------------------------------------------------
#   With static libraries, position-independent code prppertie needs to be
#   explicitly enabled.
#----------------------------------------------------------------------------
set_target_properties(vectors PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(${PROJECT_NAME} "main.cpp")

#----------------------------------------------------------------------------
#   Separate compilation and linking allow components of a CUDA program to be
#   compiled into separate objects.
#----------------------------------------------------------------------------
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(${PROJECT_NAME} PRIVATE vectors)
