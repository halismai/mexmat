# use the system default c++ compiler
# you can change your default on debian/ubuntu with
# sudo update-alternative --config c++
# or manually change to a version of g++ that is compatible with matlab
CXX = c++

TOP_DIR = $(dir $(lastword $(MAKEFILE_LIST)))

#
# edit this to your Matlab installation path if needed
#
#MATLAB_ROOT = $(shell $(MEX) -v 2>&1 | sed -n 's/.*MATLAB *= *\(.*\)/\1/gp')
MATLAB_ROOT = /usr/local/MATLAB/R2014a/

###### the rest should work without modifications ######
MEX 	    = mex
MEX_SUFFIX  = $(shell mexext)
MEX_FLAGS   = -cxx CC='$(CXX)' CXX='$(CXX)' LD='$(CXX)'
MEX_OUT_DIR = .

CXX_FLAGS = -I. -I../                                         \
	    -I$(MATLAB_ROOT)/extern/include         	      \
            -D_DEBUG_
# other stuff
CXX_FLAGS += -Wall -fPIC -fopenmp -mtune=native -msse2 -O3 -std=c++11
CXX_FLAGS += -I/usr/include/mpi/ -funroll-loops -ftree-vectorize
CXX_FLAGS += -Wno-unused-local-typedefs

# set this to 1 to use uninitialized arrays in mex
CXX_FLAGS += -DMEXMAT_NO_INIT_ARRAY=0

# optional
CXX_FLAGS += `pkg-config --cflags eigen3`
CXX_FLAGS += -DMEXMAT_WITH_EIGEN

LD_FLAGS  += -lm -lgomp

