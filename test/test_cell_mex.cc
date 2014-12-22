#include "mexmat.h"

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, mxArray const* prhs[])
{
  //
  // example of using mex::Cell
  //

  // create a cell array
  mxArray* cell_array = mex::newMexCell(3,4);
  //
  // now you can use cell_array via mex
  mex::printf("m: %d, n: %d\n", (int) mxGetM(cell_array), (int) mxGetN(cell_array));

  // or, via the class wrapper
  mex::Cell C(3, 4);

  for(mwSize i = 0; i < 3*4; ++i)
      C.set( i,  mex::newMexMatrix(3,4) );

  const mxArray* cell_element = C(0,0);
  mexPrintf("cell_element size %dx%d\n",
              (int) mxGetM(cell_element),
              (int) mxGetN(cell_element));
  // or
  // cell_array = C[5]; // linear index operator


  //
  // TODO the class needs accessors for dimensions, etc and the ability to
  // integrate with mex::Mat (e.g. implicit conversion constructor in mex::Mat)
  //
}

