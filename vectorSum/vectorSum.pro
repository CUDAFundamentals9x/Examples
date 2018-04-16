QT -= gui

OBJECTS_DIR = obj

#----------------------------------------------------------------------------
# GPU gencode
#----------------------------------------------------------------------------
GENCODE = arch=compute_30,code=sm_30
MOC_DIR = moc
CONFIG += c++11 console
CONFIG -= app_bundle

#----------------------------------------------------------------------------
# Compiler will emit warnings for deprecated features.
#----------------------------------------------------------------------------
DEFINES += QT_DEPRECATED_WARNINGS
DEFINES += _USE_MATH_DEFINES

#----------------------------------------------------------------------------
# Basic compiler flags
#----------------------------------------------------------------------------
QMAKE_CXXFLAGS += -msse -msse2 -msse3
VPATH += ./
SOURCES += main.cpp
HEADERS += vectorsum.h
INCLUDEPATH += ./
DESTDIR = ./

#----------------------------------------------------------------------------
# Define to avoid confilcs in minwindef.h and helper_math.h
#----------------------------------------------------------------------------
DEFINES += NOMINMAX

#----------------------------------------------------------------------------
# Set out cuda sources
#----------------------------------------------------------------------------
CUDA_SOURCES = vectorsum.cu

#----------------------------------------------------------------------------
# Path to cuda SDK install
#----------------------------------------------------------------------------
CUDA_DIR = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1"

#----------------------------------------------------------------------------
# Cuda include paths
#----------------------------------------------------------------------------
INCLUDEPATH += $$CUDA_DIR/include

#----------------------------------------------------------------------------
# Path to cuda toolkit install
#----------------------------------------------------------------------------
CUDA_SDK = "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v9.1"

#----------------------------------------------------------------------------
# Path to helper functions from NVIDIA
#----------------------------------------------------------------------------
win32:INCLUDEPATH += $$CUDA_SDK\common\inc

#----------------------------------------------------------------------------
# CUDA libs
#----------------------------------------------------------------------------
QMAKE_LIBDIR += $$CUDA_DIR\lib\x64
QMAKE_LIBDIR +=$$CUDA_SDK\common\lib\x64
LIBS += -lcudart -lcudadevrt

#----------------------------------------------------------------------------
# join the includes in a line
#----------------------------------------------------------------------------
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

#----------------------------------------------------------------------------
# nvcc flags (ptxas option verbose is always useful)
#----------------------------------------------------------------------------
NVCCFLAGS = --compiler-options  -fno-strict-aliasing --ptxas-options=-v -maxrregcount 20 --use_fast_math

#----------------------------------------------------------------------------
# On windows define Debug or Release mode
#----------------------------------------------------------------------------
CONFIG(debug, debug|release) {
# Debug
    # MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
    MSVCRT_LINK_FLAG_DEBUG = "/MDd"
    NVCCFLAGS += -D_DEBUG -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG
}
else {
# Release
    MSVCRT_LINK_FLAG_RELEASE = "/MD"
    NVCCFLAGS += -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE
}

#----------------------------------------------------------------------------
# Prepare intermediat CUDA compiler
#----------------------------------------------------------------------------
cudaIntr.input = CUDA_SOURCES

#----------------------------------------------------------------------------
# Windows object files have to be named with the .obj suffix
#----------------------------------------------------------------------------
cudaIntr.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}.obj

#----------------------------------------------------------------------------
# Sets NVCC with the GPU compute capability
#----------------------------------------------------------------------------
cudaIntr.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode $$GENCODE -dc $$NVCCFLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

#----------------------------------------------------------------------------
# Set our variable out. Obj files need to be used to create the link obj file
#----------------------------------------------------------------------------
cudaIntr.variable_out = CUDA_OBJ
cudaIntr.variable_out += OBJECTS
cudaIntr.clean = cudaIntrObj/*.o
cudaIntr.clean = cudaIntrObj/*.obj
QMAKE_EXTRA_COMPILERS += cudaIntr

#----------------------------------------------------------------------------
# Prepare the linking compiler step
#----------------------------------------------------------------------------
cuda.input = CUDA_OBJ
cuda.output = ${QMAKE_FILE_BASE}_link.obj

#----------------------------------------------------------------------------
# Sets NVCC with the GPU compute capability
#----------------------------------------------------------------------------
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode $$GENCODE -dlink ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}

QMAKE_EXTRA_COMPILERS += cuda




