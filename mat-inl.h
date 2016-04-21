/*
   This file is part of mexmat

   mexmat is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   mexmat is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MEXMAT_MEXMAT_MAT_INL_H
#define MEXMAT_MEXMAT_MAT_INL_H

#include <iostream>

// define this in build system
// #define MEX_MAT_NO_INIT 1

namespace mex {

template <typename _T>
Mat<_T>::Mat() : mx_ptr_(newEmptyMexMatrix<_T, mxREAL>()), owns_(true) {}

template <typename _T>
Mat<_T>::Mat(Mat&& m) : mx_ptr_(m.mx_ptr_), owns_(m.owns_) {
  // TODO check this will not break Matlab's internal memory managment
  mex::printf("move\n");
}

template <typename _T>
Mat<_T>::Mat(mwSize m, mwSize n, mxComplexity c) :
#if MEXMAT_NO_INIT_ARRAY
    mx_ptr_(newMexMatrixNoInit<_T>(m,n,c)),
#else
    mx_ptr_(newMexMatrix<_T>(m,n,c)),
#endif
    owns_(true) {}

template <typename _T>
Mat<_T>::Mat(mwSize m, mwSize n, mwSize k, mxComplexity c) :
    mx_ptr_(newMexArray<_T>(m,n,k,c)), owns_(true) {}

template <typename _T>
Mat<_T>::Mat(const std::vector<_T>& vals) :
    mx_ptr_(newMexMatrix<_T>(vals.size(),1)), owns_(true)
{ memcpy(this->data(), vals.data(), sizeof(_T)*vals.size()); }

template <typename _T>
Mat<_T>::Mat(mwSize nd, const mwSize* dims) :
    mx_ptr_(newMexArray<_T>(nd,dims)), owns_(true) {}

template <typename _T>
Mat<_T>::Mat(const Mat<_T>& m) :
    mx_ptr_(mxDuplicateArray(m.mxData())), owns_(true) {}

template <typename _T>
Mat<_T>& Mat<_T>::operator=(const Mat<_T>& m)
{
  if(this != &m) {
    free();
    mx_ptr_ = mxDuplicateArray(m.mxData());
    owns_ = true;
  }

  return *this;
}

template <typename _T>
void Mat<_T>::free()
{
  if(owns_) {
    mex::destroyArray(mx_ptr_);
    owns_ = false;
  }
}

template <typename __T> inline
std::ostream& operator<<(std::ostream& os, const Mat<__T>& m)
{
  for(mwSize i=0; i<m.rows(); ++i) {
    for(mwSize j=0; j<m.cols(); ++j)
      os << m(i, j) << "  ";
    os << "\n";
  }

  return os;
}

inline std::ostream& operator<<(std::ostream& os, const Mat<char>& m)
{
  os << getString(m);
  return os;
}

}; // mex

#endif // MEXMAT_MEXMAT_MAT_INL_H

