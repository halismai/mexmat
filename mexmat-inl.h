/*
   Copyright (C) 2013-2014

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

#ifndef MEXMAT_MEXMAT_INL_H
#define MEXMAT_MEXMAT_INL_H

//
// implementation is kept in a header to ease compilation
//

namespace mex {

inline std::string getString(const mxArray* a)
{
  assertType<char>(a);

  const mwSize N = length(a) + 1;
  char* buf = mex::malloc<char>(N);
  if(0 != mxGetString(a, buf, N)) {
    mex::free(buf);
    mex::error("failed to get string");
  }

  std::string ret(buf);
  mex::free(buf);

  return ret;
}

inline mxArray* createString(const std::string& string)
{
  return mxCreateString(string.c_str());
}

template <typename T> inline
mxArray* valueToMxArray(const T& value)
{
  static_assert( std::is_arithmetic<T>::value,
                "valueToMxArray: value must be arithmetic");

  mex::Mat<T> ret(1,1);
  ret[0] = value;
  return ret.release();
}

template <> inline mxArray* valueToMxArray<std::string>(const std::string& s)
{
  return createString(s);
}

inline std::string id2string(mxClassID id)
{
  switch(id)
  {
    case mxUNKNOWN_CLASS:   return "unkown";
    case mxCELL_CLASS:      return "cell";
    case mxSTRUCT_CLASS:    return "struct";
    case mxLOGICAL_CLASS:   return "logical";
    case mxCHAR_CLASS:      return "char";
    case mxVOID_CLASS:      return "void";
    case mxDOUBLE_CLASS:    return "double";
    case mxSINGLE_CLASS:    return "single";
    case mxINT8_CLASS:      return "int8";
    case mxUINT8_CLASS:     return "uint8";
    case mxINT16_CLASS:     return "int16";
    case mxUINT16_CLASS:    return "uint16";
    case mxINT32_CLASS:     return "int32";
    case mxUINT32_CLASS:    return "uint32";
    case mxINT64_CLASS:     return "int64";
    case mxUINT64_CLASS:    return "uint64";
    case mxFUNCTION_CLASS:  return "function";
    case mxOPAQUE_CLASS:    return "opaque";
    case mxOBJECT_CLASS:    return "object";
    default:                return "bad class";
  }
}

}; // mex
#endif // MEXMAT_MEXMAT_INL_H

