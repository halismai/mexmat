#include "mexmat.h"
#if !defined(MEXMAT_WITH_EIGEN)
void mexFunction(int, mxArray*[], int, mxArray const* [])
{
  mexError("compile with Eigen\n");
}

#else

#include <Eigen/Dense>
#include <Eigen/LU>

template <typename Scalar>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

//
// Example of solving a linear system via Eigen
//
static const char* USAGE = "x = test_eigen_mex(A, b)";

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, mxArray const* prhs[])
{
  mex::nargchk(2, 2, nrhs, USAGE);

  const mex::Mat<double> A(prhs[0]);
  const mex::Mat<double> b(prhs[1]);

  //
  // make sure the we have more equations than unknowns
  //
  mex::massert( A.rows() <= A.cols() );
  mex::massert( A.rows() == b.rows() );

  const auto A_eigen = A.toEigen();
  const auto b_eigen = b.toEigen();

  Matrix<double> x = A_eigen.fullPivLu().solve(b_eigen);

  mex::Mat<double> ret(x);
  plhs[0] = ret.release();
}
#endif
