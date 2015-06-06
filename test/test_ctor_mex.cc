#include <mexmat.h>

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, mxArray const* prhs[])
{

  Eigen::Matrix<double,3,1> X = Eigen::Matrix<double,3,1>::Random();
  mex::Struct ret({"X"});

  ret.set("X", mex::Mat<double>(X));
  plhs[0] = ret.release();
}
