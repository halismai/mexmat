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

#ifndef MEXMAT_MEXMAT_CV_H
#define MEXMAT_MEXMAT_CV_H

#if defined(MEXMAT_WITH_OPENCV)

#include <opencv2/core/core.hpp>

namespace mex {

template <typename T> inline
mex::Mat<T> cv2mex(const cv::Mat& src)
{
  // conversion based on cv/eigen
  if(src.type() != cv::DataType<T>::type)
    mexError("incompatible type in conversion\n");
  if(src.channels() > 1)
    mexError("conversion is support for single channel images only\n");

  mex::Mat<T> dst(src.rows, src.cols);
  cv::Mat _dst(src.cols, src.rows, cv::DataType<T>::type, (void*) dst.data());

  cv::transpose(src, _dst);
  return dst;
}

template <typename T> inline
cv::Mat mex2cv(const mex::Mat<T>& src)
{
  cv::Mat dst(src.rows(), src.cols(), cv::DataType<T>::type);

  cv::Mat _src(src.cols(), src.rows(), cv::DataType<T>::type, (void*) src.data());
  cv::transpose(_src, dst);

  return dst;
}

inline mxArray* cv2mexarray(const cv::Mat& src)
{
  switch(src.type()) {

    case cv::DataType<double>::type:
      {
        return cv2mex<double>(src).release();
      } break;

    case cv::DataType<float>::type:
      {
        return cv2mex<float>(src).release();
      } break;

    case cv::DataType<uint8_t>::type:
      {
        return cv2mex<uint8_t>(src).release();
      } break;

    case cv::DataType<uint16_t>::type:
      {
        return cv2mex<uint16_t>(src).release();
      } break;

    case cv::DataType<int16_t>::type:
      {
        return cv2mex<int16_t>(src).release();
      } break;

    case cv::DataType<uint32_t>::type:
      {
        return cv2mex<uint32_t>(src).release();
      } break;

    case cv::DataType<int32_t>::type:
      {
        return cv2mex<int32_t>(src).release();
      } break;
    default:
      {
        mexError("unhandled type '%d'\n", src.type());
        return NULL;
      }
  }
}


}; // mex

#endif // MEXMAT_WITH_OPENCV

#endif // MEXMAT_MEXMAT_CV_H

