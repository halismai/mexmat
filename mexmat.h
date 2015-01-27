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

#ifndef MEXMAT_MEXMAT_H
#define MEXMAT_MEXMAT_H

#include "mexmat_config.h"

#include <cstdint>
#include <cstring>
#include <cstdarg>
#include <typeinfo>
#include <string>
#include <vector>
#include <ostream>
#include <cstdio>
#include <map>

namespace mex {

template <typename> class Mat;

/** Issues an error in matlab. Will terminate the program */
static inline void error(const std::string& msg) { mexError(msg.c_str()); }

/** Issues a warning in matlab. Does not terminate the program */
static inline void warning(const std::string& msg) { mexWarning(msg.c_str()); }

/** printf */
static inline void printf(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);
}

/** asserts if cond is false, the assert will issue an error in Matlab */
static inline void massert(bool cond, const std::string& msg="") {
  if(!cond) mexError(msg.c_str());
}

/** asserts if number of args n is less than lo or greater than hi */
inline void nargchk(int lo, int hi, int n, const std::string& msg="") {
  massert(n>=lo && n<=hi, msg.empty() ? "wrong number of args" : msg);
}

/** \return a string from the array. mex class must be mxCHAR_CLASS, otherwise
 * the function asserts */
std::string getString(const mxArray* a);

template <typename _T> inline _T* malloc(mwSize n) {
  return static_cast<_T*>(mxMalloc(n*sizeof(_T)));
}

template <typename _T> inline _T* calloc(mwSize n) {
  return static_cast<_T*>(mxCalloc(n,sizeof(_T)));
}

template <typename _T> inline _T* realloc(_T* p, mwSize n) {
  return static_cast<_T*>(mxRealloc(p, n*sizeof(_T)));
}

template <typename _T> inline void free(_T* p) { mxFree(p); }

inline void* getData(const mxArray* a) { return mxGetData(a); }
inline void* data(const mxArray *a) { return getData(a); }

//
// various utilties
//
inline mwSize numel (const mxArray* a) { return mxGetNumberOfElements(a); }
inline mwSize rows  (const mxArray* a) { return mxGetM(a); }
inline mwSize cols  (const mxArray* a) { return mxGetN(a); }
inline mwSize ndims (const mxArray* a) { return mxGetNumberOfDimensions(a); }

inline const mwSize* dims(const mxArray* a) { return mxGetDimensions(a); }

/**
 * \return a dimension based on D (zero-based)
 *
 * E.g. to get the row of a matrix dim<0>(a) and the cols dim<1>(a).
 * If you have a 2D image, for example, dim<2>(a) is the number of channels
 *
 * If the matrix does not have the specified dimension, the function returns 0
 */
template <mwSize _D> inline mwSize dim(const mxArray* a) {
  return (ndims(a) > _D) ? dims(a)[_D] : 0;
}

inline bool isComplex (const mxArray* a) { return mxIsComplex(a); }
inline bool isLogical (const mxArray *a) { return mxIsLogical(a); }
inline bool isChar    (const mxArray* a) { return mxIsChar(a); }
inline bool isDouble  (const mxArray* a) { return mxIsDouble(a); }
inline bool isSingle  (const mxArray* a) { return mxIsSingle(a); }
inline bool isNumeric (const mxArray* a) { return mxIsNumeric(a); }
inline bool isNumber  (const mxArray* a) { return mxIsNumeric(a); }
inline bool isCell    (const mxArray* a) { return mxIsCell(a); }
inline bool isStruct  (const mxArray* a) { return mxIsStruct(a); }
inline bool isOpaque  (const mxArray* a) { return mxIsOpaque(a); }
inline bool isFnHandle(const mxArray* a)  { return mxIsFunctionHandle(a); }
inline bool isClass   (const mxArray* a, const std::string & name)  {
  return mxIsClass(a,name.c_str());
}
inline bool isFloatingPoint(const mxArray* a) {
  return isDouble(a) || isSingle(a);
}
inline bool isReal(const mxArray* a) {
  return isNumeric(a) && isFloatingPoint(a);
}
inline bool isVector(const mxArray* a) {
  return (rows(a)==1 && cols(a) > 1) || (cols(a)==1 && rows(a) > 1);
}
inline bool isScalar(const mxArray* a) { return rows(a)==1 && (1==cols(a)); }

inline mwSize length(const mxArray* a) { return std::max(rows(a), cols(a)); }

inline std::string className(const mxArray* a) { return mxGetClassName(a); }
inline mxClassID classId(const mxArray* a) { return mxGetClassID(a); }
inline mxClassID id(const mxArray* a) { return classId(a); }

class _uknown_type {};

template<typename _T = _uknown_type> class traits_ {
public:
 static const mxClassID type         = mxUNKNOWN_CLASS;
 static const mxComplexity complex_t = mxCOMPLEX;
 static const mxComplexity real_t    = mxREAL;
 static std::string string() { return "unkown"; }
}; // traits_

template<> class traits_<bool> {
public:
 static const mxClassID type = mxLOGICAL_CLASS;
 static std::string string() { return "bool"; }
}; // traits_

template<> class traits_<char> {
public:
 static const mxClassID type = mxCHAR_CLASS;
 static std::string string() { return "char"; }
}; // traits_

template<> class traits_<int8_t> {
public:
 static const mxClassID type = mxINT8_CLASS;
 static std::string string() { return "int8_t"; }
}; // traits_

template<> class traits_<int16_t> {
public:
 static const mxClassID type = mxINT16_CLASS;
 static std::string string() { return "int16_t"; }
}; // traits_

template<> class traits_<int32_t> {
public:
 static const mxClassID type = mxINT32_CLASS;
 static std::string string() { return "int32_t"; }
}; // traits_

template<> class traits_<int64_t> {
public:
 static const mxClassID type = mxINT64_CLASS;
 static std::string string() { return "int64_t"; }
}; // traits_

template<> class traits_<uint8_t> {
public:
 static const mxClassID type = mxUINT8_CLASS;
 static std::string string() { return "uint8_t"; }
}; // traits_

template<> class traits_<uint16_t> {
public:
 static const mxClassID type = mxUINT16_CLASS;
 static std::string string() { return "uint16_t"; }
}; // traits_

template<> class traits_<uint32_t> {
public:
 static const mxClassID type = mxUINT32_CLASS;
 static std::string string() { return "uint32_t"; }
}; // traits_

template<> class traits_<uint64_t> {
public:
 static const mxClassID type = mxUINT64_CLASS;
 static std::string string() { return "uint64_t"; }
}; // traits_

template<> class traits_<float> {
public:
 static const mxClassID type = mxSINGLE_CLASS;
 static std::string string() { return "single"; }
}; // traits_

template<> class traits_<double> {
public:
 static const mxClassID type = mxDOUBLE_CLASS;
 static std::string string() { return "double"; }
}; // traits_

template <typename _T=_uknown_type> class mex_traits_ {
  typedef void type;
}; // mex_traits_

/** asserts (causes a matlab error) if the underlying data type of a is not T */
template <typename _T> inline void assertType(const mxArray* a) {
  mex::massert(traits_<_T>::type == classId(a),
               "expcting class `" + traits_<_T>::string() + "', but got a `" +
               className(a) + "' instead.");
}

inline void assertNumber(const mxArray* a) {
  mex::massert(isScalar(a), "expecting a scalar value");
}
inline void assertLogical(const mxArray* a) {
  mex::massert(isLogical(a), "expecting a logical value");
}
inline void assertScalar(const mxArray* a) { assertNumber(a); }

inline void assertSize(const mxArray* a, mwSize m, mwSize n,
                       const std::string& msg="input args size mismatch") {
  massert(rows(a)==m && cols(a)==n, msg);
}

inline void assertSize(const mxArray* a, mwSize m, mwSize n, mwSize k,
                        const std::string& msg) {
  massert(rows(a)==m && cols(a)==n && dim<2>(a)==k, msg);
}

template <typename T1, typename T2>
inline void assertSize(const mex::Mat<T1>&, const mex::Mat<T2>&,
                       const std::string& msg="input args size mismatch");

/**
 * \return a number from 'a'
 * the function does not check for narrowing type conversion
 *
 * if check is true, the function will assert if the array is not a scalar
 */
template <typename _T> inline _T getNumber(const mxArray* a, bool check=false) {
  if(check) { assertNumber(a); }
  return static_cast<_T>(mxGetScalar(a));
}

template <typename _T> inline _T getScalar(const mxArray* a, bool check=false) {
   return getNumber<_T>(a, check);
}

inline bool getLogical(const mxArray* a, bool check = false) {
  if(check) { assertLogical(a); }
  return static_cast<bool>(*mxGetLogicals(a));
}

namespace internal {

inline mwSize error1(const std::string& msg) { mex::error(msg); return 0; }

inline static constexpr
mwSize require_valid_ind(mwSize nrows, mwSize ncols, mwSize r, mwSize c) {
  return (r>=nrows) || (c>=ncols) ? error1("index out of bounds") : 0;
}

inline static constexpr
mwSize idx_col_major(mwSize nrows, mwSize ncols, mwSize r, mwSize c) {
  return require_valid_ind(nrows,ncols,r,c), (r + c*nrows);
}

inline static constexpr
mwSize idx_col_major(mwSize nrows, mwSize r, mwSize c) {
  return r + c*nrows;
}

class size {
 public:
  size() : rows_(0), cols_(0) {}
  size(const mxArray* a) : rows_(a?rows(a):0), cols_(a?cols(a):0) {}
  size(mwSize m, mwSize n) : rows_(m), cols_(n) {}
  inline mwSize sub2ind(mwSize r, mwSize c) const {
    return internal::idx_col_major(rows_, cols_, r, c);
  }

 protected:
  mwSize rows_;
  mwSize cols_;
}; // size
}; // internal

template <typename _T = double, mxComplexity C = mxREAL> inline
mxArray* newMexMatrix(mwSize m, mwSize n) {
  	return mxCreateNumericMatrix(m,n,traits_<_T>::type,C);
}

template <typename _T = double, mxComplexity C = mxREAL> inline
mxArray* newMexMatrixNoInit(mwSize m, mwSize n) {
  mxArray* data = newMexMatrix<_T,C>(0,0);
  mwSize dims[] = {m, n};
  mxSetDimensions(data, dims, 2);
  mxSetData(data, mxMalloc(m*n*sizeof(_T)));
  return data;
}

mxArray* newMexCell(mwSize nr, mwSize nc) { return mxCreateCellMatrix(nr,nc); }

template <typename _T = double> inline
mxArray* newMexMatrix(mwSize m, mwSize n, mxComplexity c) {
  return mxCreateNumericMatrix(m,n,traits_<_T>::type, c);
}

template <typename _T = double, mxComplexity c = mxREAL> inline
mxArray* newEmptyMexMatrix() {
  return mxCreateNumericMatrix(0, 0, traits_<_T>::type, c);
}

template <typename _T = double> inline
mxArray* newMexMatrixNoInit(mwSize m, mwSize n, mxComplexity c) {
  mxArray* data = newMexMatrix<_T>(0, 0, c);
  mwSize dims[] = {m, n};
  mxSetDimensions(data, dims, 2);
  mxSetData(data, mxMalloc(m*n*sizeof(_T)));
  return data;
}

template <typename _T = double, mxComplexity C = mxREAL> inline
mxArray* newMexArray(mwSize m, mwSize n, mwSize k) {
  mwSize dims[3] = {m,n,k};
  return mxCreateNumericArray(3, dims, traits_<_T>::type, C);
}

template <typename _T = double> inline
mxArray* newMexArray(mwSize m, mwSize n, mwSize k, mxComplexity c = mxREAL) {
  mwSize dims[3] = {m,n,k};
  return mxCreateNumericArray(3, dims, traits_<_T>::type, c);
}

template <typename _T = double, mxComplexity C = mxREAL> inline
mxArray* newMexArray(const std::vector<mwSize>& dims) {
  return mxCreateNumericArray(dims.size(), &dims[0], traits_<_T>::type, C);
}

template <typename _T = double, mxComplexity C = mxREAL> inline
mxArray* newMexMatrix(mwSize ndims, const mwSize* dims) {
  return mxCreateNumericArray(ndims, dims, traits_<_T>::type, C);
}

template <typename _T = double> inline
mxArray* newMexMatrix(mwSize ndims, const mwSize* dims, mxComplexity c=mxREAL) {
  return mxCreateNumericArray(ndims, dims, traits_<_T>::type, c);
}


inline void destroyArray(mxArray*& a) {
  if(a) { mxDestroyArray(a); a = NULL; }
}

inline int callMatlab(int nlhs, mxArray* plhs[], int nrhs, mxArray* prhs[],
                       const std::string& name) {
  return mexCallMATLAB(nlhs, plhs, nrhs, prhs, name.c_str());
}

inline mxArray* callMatlabTrap(int nlhs, mxArray* plhs[],
                               int nrhs, mxArray* prhs[],
                               const std::string& fn) {
  return mexCallMATLABWithTrap(nlhs, plhs, nrhs, prhs, fn.c_str());
}

inline int eval(const std::string& expr) { return mexEvalString(expr.c_str()); }

inline void makePersistent(mxArray* a) { mexMakeArrayPersistent(a); }
inline void makePersistent(void* a)    { mexMakeMemoryPersistent(a); }

inline void lock() { mexLock(); }
inline void unlock() { mexUnlock(); }

inline int atexit(void (*fn)(void)) { return mexAtExit(fn); }

std::string id2string(mxClassID);

template <typename _T = double>
class Mat {
 public:
  typedef _T value_type;
  //typedef _T Scalar;

 public:
  /** empty ctor creates an empty array */
  Mat();

  /** move ctor */
  Mat(Mat&&);

  /** create a matrix with size m x n and complexity C */
  Mat(mwSize m, mwSize n, mxComplexity c = mxREAL);

  /** create a matrix with size m x n x k and complexity C */
  Mat(mwSize m, mwSize n, mwSize k, mxComplexity c = mxREAL);

  /** does not allocate memory, just a wrapper around non-const mxArray */
  template <class _MA,
   typename std::enable_if<
       std::is_same<_MA,mxArray>::value,int>::type=0> Mat(_MA* a) :
           mx_ptr_(a), owns_(false) { assertType<_T>(a); }

  /** does not allocate memory, wrapper around a cost mxArray */
  template <class _MA,
   typename std::enable_if<
       std::is_same<_MA,const mxArray>::value,int>::type=0> Mat(_MA* a) :
           mx_ptr_(const_cast<mxArray*>(a)), owns_(false) { assertType<_T>(a); }

  /** creates a colmun vector from a std::vector */
  Mat(const std::vector<_T>&);

  /** creates a multi dimensional array */
  Mat(mwSize ndims, const mwSize* dims);

  /** copy constructor */
  Mat(const Mat&);

  /** assigment */
  Mat& operator=(const Mat&);

  /** assignment */
  Mat& operator=(Mat);

  /** destructor, frees memory if owned */
  ~Mat() { free(); }

#if defined(MEXMAT_WITH_EIGEN)
  template <class __EigenMatrix>
  Mat(const __EigenMatrix& mat) :
      Mat<typename __EigenMatrix::Scalar>(mat.rows(), mat.cols())
  {
    memcpy( this->data(), mat.data(), mat.rows() * mat.cols()
            * sizeof(typename __EigenMatrix::Scalar));
  }
#endif

 public:  /* static stuff */
  template <typename __T=double, mxComplexity __C = mxREAL> inline static
  Mat<__T> Matrix(mwSize m, mwSize n) { return Mat<__T>(m, n, __C); }

  template <typename __T=double, mxComplexity __C = mxREAL> inline static
  Mat<__T> Matrix(mwSize m, mwSize n, mwSize k) { return Mat<__T>(m,n,k,__C); }

  template <typename __T=double, mxComplexity __C = mxREAL> inline static
  Mat<__T> ColVector(mwSize m) { return Mat<__T>(m,1,__C); }

  template <typename __T=double, mxComplexity __C = mxREAL> inline static
  Mat<__T> RowVector(mwSize n) { return Mat<__T>(1,n,__C); }

  template <typename __T=double, mxComplexity __C = mxREAL> inline static
  Mat<__T> Scalar(__T v=static_cast<__T>(0)) {
    Mat<__T> ret(1,1,__C); ret[0] = v; return ret;
  }

 public:
  inline       mxArray* mxData()       { return mx_ptr_; }
  inline const mxArray* mxData() const { return mx_ptr_; }

  inline       void* data()       { return mex::getData(mx_ptr_); }
  inline const void* data() const { return mex::getData(mx_ptr_); }

  inline       _T* ptr()       { return static_cast<_T*>(data()); }
  inline const _T* ptr() const { return static_cast<const _T*>(data()); }


  inline       _T* col(int i)       { return &(this->operator()(0,i)); }
  inline const _T* col(int i) const { return &(this->operator()(0,i)); }

  inline mwSize length() const { return mex::length(mx_ptr_); }
  inline mwSize size() const { return length(); }
  inline mwSize rows()   const { return mex::rows(mx_ptr_); }
  inline mwSize cols()   const { return mex::dim<1>(mx_ptr_); }
  inline mwSize depth()  const { return mex::dim<2>(mx_ptr_); }
  inline mwSize ndims()  const { return mex::ndims(mx_ptr_); }
  template <mwSize _D> mwSize dim() const { return mex::dim<_D>(mx_ptr_); }

  inline bool empty() const { return 0==length(); }

  inline std::string className() const { return mex::className(mx_ptr_); }
  inline mxClassID classId() const { return mex::classId(mx_ptr_); }

  inline mwSize sub2ind(mwSize r, mwSize c) const { return r + c*rows(); }
  inline mwSize sub2ind(mwSize r, mwSize c, mwSize k) const {
    return (r*cols()+c)*depth()+k ;
  }

  /** relinquishes ownership of the pointer */
  mxArray* release() { owns_=false; return mx_ptr_; }

 public:
  inline       _T& operator()(int r, int c)       { return ptr()[sub2ind(r,c)]; }
  inline const _T& operator()(int r, int c) const { return ptr()[sub2ind(r,c)]; }

  inline       _T& operator()(int r, int c, int k) {
    return ptr()[sub2ind(r,c,k)]; }
  inline const _T& operator()(int r, int c, int k) const {
    return ptr()[sub2ind(r,c,k)];
  }

  inline       _T& operator[](int i)       { return ptr()[i]; }
  inline const _T& operator[](int i) const { return ptr()[i]; }

  operator       _T*()       { return ptr(); }
  operator const _T*() const { return ptr(); }

  operator       _T()       { return *ptr(); }
  operator const _T() const { return *ptr(); }

  operator const  mxArray*() const { return mx_ptr_; }
  //operator      mxArray*()       { return mx_ptr_; }  // const cast only

#if defined(MEXMAT_WITH_EIGEN)
  template <
     typename __T,
     int __Rows = Eigen::Dynamic,
     int __Cols = Eigen::Dynamic,
     int __Opts = Eigen::AutoAlign | Eigen::ColMajor>
  using EigenMatrix = Eigen::Matrix<__T, __Rows, __Cols, __Opts>;

  template <
   typename __T,
   int __Rows = Eigen::Dynamic,
   int __Cols = Eigen::Dynamic,
   int __Opts = Eigen::AutoAlign | Eigen::ColMajor>
  using EigenMap = Eigen::Map< EigenMatrix<__T, __Rows, __Cols, __Opts> >;

  template <
   typename __T,
   int __Rows = Eigen::Dynamic,
   int __Cols = Eigen::Dynamic,
   int __Opts = Eigen::AutoAlign | Eigen::ColMajor>
  using EigenConstMap = Eigen::Map< const EigenMatrix<__T, __Rows, __Cols, __Opts> >;

  // assume col order in Eigen
  inline EigenMap<_T> toEigen() {
    return EigenMap<_T>( ptr(), rows(), cols() );
  }

  inline EigenConstMap<_T> toEigen() const {
    return EigenConstMap<_T>(ptr(), rows(), cols());
  }
#endif

  template <typename __T>
  friend std::ostream& operator<<(std::ostream&, const Mat<__T>&);
  friend std::ostream& operator<<(std::ostream&, const Mat<char>&);

 protected:
  void free();

 protected:
  mxArray* mx_ptr_; /** mxArray type */
  bool     owns_; /** true if the class owns the data */
}; // Mat

template <typename T1, typename T2> inline
void assertSize(const mex::Mat<T1>& a, const mex::Mat<T2>& b,
                const std::string& msg) {
  mex::massert( a.rows()==b.rows() && a.cols()==b.cols(), msg );
}


/** Wrapper on cell */
class Cell {
 public:
  Cell() : mx_ptr_(NULL), owns_(false) {}

  ~Cell() { free(); }

  template <class C, typename std::enable_if<
   (std::is_same<C, mxArray>::value),int>::type=0> Cell(C* c) :
       mx_ptr_(c), owns_(false) { massert(isCell(mx_ptr_)); }

  template <class C, typename std::enable_if<
   (std::is_same<C, const mxArray>::value),int>::type=0> Cell(C* c) :
       mx_ptr_(const_cast<mxArray*>(c)), owns_(false) {
         massert(isCell(mx_ptr_));
  }

  Cell(mwSize m, mwSize n) : mx_ptr_(newMexCell(m,n)), owns_(true) {}

  // move
  Cell(Cell&& c) : mx_ptr_(c.mx_ptr_), owns_(c.owns_) { c.owns_=false; }

  inline const mxArray* operator()(mwIndex i, mwIndex j) const {
    mwIndex a[] = {i,j};
    return mxGetCell(mx_ptr_, mxCalcSingleSubscript(mx_ptr_, 2, a));
  }

  inline mxArray* operator()(mwIndex i, mwIndex j) {
    mwIndex a[] = {i,j};
    return mxGetCell(mx_ptr_, mxCalcSingleSubscript(mx_ptr_, 2, a));
  }

  inline const mxArray* operator[](mwIndex ii) const {
    return mxGetCell(mx_ptr_, ii);
  }

  inline mxArray* operator[](mwIndex ii) {
    return mxGetCell(mx_ptr_, ii);
  }

  inline void set(mwIndex ii, mxArray* a) { mxSetCell(mx_ptr_, ii, a); }

  inline mxArray* release() { owns_=false; return mx_ptr_; }

  inline mwSize length() const { return mex::length(mx_ptr_); }

 protected:
  inline void free() {
    if(owns_) {
      mex::destroyArray(mx_ptr_);
      owns_ = false;
    }
  }

 protected:
  mxArray* mx_ptr_;
  bool owns_;
}; // Cell

/** Wrapper for struct */
class Struct
{
 public:
  Struct() : mx_ptr_(NULL), owns_(false) {}

  ~Struct() { free(); }

  template <class C,
  typename std::enable_if<(std::is_same<C, mxArray>::value), int>::type=0>
      Struct(C* c) : mx_ptr_(c), owns_(false)
  {
    massert(isStruct(mx_ptr_));
    setData();
  }

  template <class C,
  typename std::enable_if<(std::is_same<C, const mxArray>::value),int>::type=0>
      Struct(C* c) : mx_ptr_(const_cast<mxArray*>(c)), owns_(false)
  {
    massert(isStruct(mx_ptr_));
    setData();
  }

  Struct(const std::vector<std::string>& field_names, int rows=1, int cols=1) :
      owns_(true)
  {
    const size_t nfields = field_names.size();
    const char* names[nfields];
    for(size_t i = 0; i < nfields; ++i)
      names[i] = field_names[i].c_str();

    mx_ptr_ = mxCreateStructMatrix(rows, cols, nfields, names);
    setData();
  }


  const mxArray* operator[](const std::string& name) const
  {
    try {
      return _data[0].at(name);
    } catch( ... ) {
      mexError("invalid name: `%s'\n", name.c_str());
    }

    return NULL;
  }

  const mxArray* getField(const std::string& name, mwIndex ind = 0) const
  {
    const mxArray* ret = mxGetField(mx_ptr_, ind, name.c_str());
    if(!ret)
      mexError("invalid name: `%s'\n", name.c_str());

    return ret;
  }

  mxArray* getField(const std::string& name, mwIndex ind = 0)
  {
    mxArray* ret = mxGetField(mx_ptr_, ind, name.c_str());
    if(!ret)
      mexError("invalid name: `%s'\n", name.c_str());

    return ret;
  }

  mxArray* release()
  {
    owns_ = false;
    return mx_ptr_;
  }

  inline Struct& set(const std::string& fname, mxArray* val, mwIndex ind = 0)
  {
    mxSetField(mx_ptr_, ind, fname.c_str(), val);
    return *this;
  }

  template <typename T>
  inline Struct& set(const std::string& fname, mex::Mat<T>& m, mwSize ind =0)
  {
    return set(fname, m.release(), ind);
  }

  template <typename T> inline
  Struct& operator()(const std::string& fname, mex::Mat<T>& m, mwSize ind = 0)
  {
    return set(fname, m, ind);
  }

  inline mwSize length() const
  {
    return mex::length(mx_ptr_);
  }

 protected:
  inline void free()
  {
    if(owns_) {
      destroyArray(mx_ptr_);
      owns_ = false;
    }
  }

  inline void setData()
  {
    mwSize n = mex::length( mx_ptr_ );
    _data.resize( n );

    int num_fields = mxGetNumberOfFields(mx_ptr_);
    _field_names.resize( num_fields );

    for(int f = 0; f < num_fields; ++f) {
      const char* name = mxGetFieldNameByNumber(mx_ptr_, f);
      _field_names[f] = std::string(name);
    }

    for(mwSize i = 0; i < n; ++i) {
      for(int f = 0; f < num_fields; ++f) {
        const mxArray* field = mxGetFieldByNumber(mx_ptr_, i, f);
        _data[i][_field_names[f]] = field;
      }
    }
  }

 protected:
  mxArray* mx_ptr_;
  bool owns_;

  //
  // an entry per element
  //
  std::vector<std::map<std::string, const mxArray*>> _data;

  std::vector<std::string> _field_names; //
}; // Struct


class Class
{
 public:
  Class() : mx_ptr_(NULL), owns_(false) {}
  ~Class() { free(); }

  template <class C, typename std::enable_if<
    (std::is_same<C, mxArray>::value), int>::type=0>
  Class(C* c) : mx_ptr_(c), owns_(false) {
    // massert(isClass(mx_ptr_)); TODO
  }

  template <class C, typename std::enable_if<
    (std::is_same<C, const mxArray>::value), int>::type=0>
  Class(C* c) : mx_ptr_(const_cast<mxArray*>(c)), owns_(false)
    {
      // massert(isClass(mx_ptr_)); TODO
    }

  inline const mxArray* operator[](const std::string& prop_name) const
  {
    // copies the data
    return mxGetProperty(mx_ptr_, 0, prop_name.c_str());
  }


 protected:
  inline void free()
  {
    if(owns_) {
      mex::destroyArray(mx_ptr_);
      owns_=false;
    }
  }

 protected:
  mxArray* mx_ptr_;
  bool owns_;
}; // Class



// Based on
// http://www.mathworks.com/matlabcentral/fileexchange/38964-example-matlab-class-wrapper-for-a-c++-class
template <typename _T>
class ClassHandle {
  static const uint8_t VALID_ID = 0xf0;
 public:
  ClassHandle(_T* p) : ptr_(p), id_(VALID_ID),
    name_(typeid(_T).name()) {}

  ~ClassHandle() {
    if(ptr_) { delete ptr_; ptr_=nullptr; id_=0; name_.clear(); }
  }

  inline bool valid() const {
    return id_ == VALID_ID && name_ == std::string(typeid(_T).name());
  }

  inline       _T* ptr()       { return ptr_; }
  inline const _T* ptr() const { return ptr_; }

  inline static mxArray* ToMex(_T* ptr)
  {
    mex::lock();
    mex::Mat<uint64_t> ret = mex::Mat<uint64_t>::Scalar(
        reinterpret_cast<uint64_t>(new ClassHandle<_T>(ptr)));
    return ret.release();
  }

  inline static ClassHandle<_T>* ToClassHandle(const mxArray* a)
  {
    mex::assertNumber(a);
    ClassHandle<_T>* ret = reinterpret_cast<ClassHandle<_T>*>(
        mex::getNumber<uint64_t>(a));

    mex::massert(ret->valid(), "not a valid class of " +
                 std::string(typeid(_T).name()));
    return ret;
  }

  // deleted stuff
  ClassHandle() = delete;
  ClassHandle(const ClassHandle&) = delete;
  ClassHandle& operator=(const ClassHandle&) = delete;

 private:
  _T* ptr_;
  uint8_t id_;
  std::string name_;
}; // ClassHandle

template <typename _T> inline mxArray* PtrToMex(_T* p) {
  return ClassHandle<_T>::ToMex(p);
}

template <typename _T> inline _T* MexToPtr(const mxArray* a) {
  return (ClassHandle<_T>::ToClassHandle(a))->ptr();
}

template <typename _T> void DeleteClass(const mxArray* a) {
  delete ClassHandle<_T>::ToClassHandle(a);
  mex::unlock();
}

}; // mex

namespace std {
template <typename T> inline
const T* begin(const mex::Mat<T>& m) { return m.ptr(); }

template <typename T> inline
T* begin(mex::Mat<T>& m) { return m.ptr(); }

template <typename T> inline
const T* end(const mex::Mat<T>& m) { return (m.ptr()+m.length()); }

template <typename T> inline
T* end(mex::Mat<T>& m) { return (m.ptr()+m.length()); }
}; // std


#include "mexmat-inl.h"
#include "mat-inl.h" // implementation of the Mat class

#endif // MEXMAT_MEXMAT_H

