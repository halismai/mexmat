#include "mexmat.h"

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, mxArray const* prhs[])
{
  mex::nargchk(1,1,nrhs,"need arg");
  const mex::Class c(prhs[0]);
  const mex::Mat<double> p(c["prop"]);

}
