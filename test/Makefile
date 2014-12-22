include ../make.inc

MEX_SRC = $(wildcard *_mex.cc)
LIB_SRC = $(filter-out $(MEX_SRC), $(wildcard *.cc,*.h))
LIB_OBJ = $(patsubst %.cc,%.o,$(LIB_SRC))

all: $(patsubst %.cc,%.$(MEX_SUFFIX), $(MEX_SRC))

%.$(MEX_SUFFIX):%.o
	$(MEX) $(MEX_FLAGS) $(LD_FLAGS) -output $(MEX_OUT_DIR)/$@ $^

%.o: %.cc
	$(CXX) $(CXX_FLAGS) -c $^

clean::
	rm -rf *.o *.$(MEX_SUFFIX)





