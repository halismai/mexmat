#include "mexmat.h"
#include <iostream>

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, mxArray const* prhs[])
{
  mex::nargchk(1, 1, nrhs, "need arg");
  const mex::Struct s( prhs[0] );

  const mxArray* a = s["name"];
  const std::string name = mex::getString(a);

  std::cout << "name: " << name << std::endl;
  std::cout << mex::getNumber<double>( s["age"] ) << std::endl;

  /*
   * verified, this will throw an error
  std::cout << mex::getNumber<int>( s["nonexistent_field"] ) << std::endl;
  */

  std::vector<std::string> names = {"f1", "f2", "f3"};
  mex::Struct ss(names);

  mex::Mat<double> f1_val(2, 3);
  ss.set("f1", f1_val);

  plhs[0] = ss.release();
}
