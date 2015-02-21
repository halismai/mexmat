# mexmat

Thin wrapper to access Matlab's external interface (MEX). The library is
header only and does not require compilation.

The current build system will look for any *_mex.cc files and link them with
Matlab's libraries. See the directory 'test/' for example programs. This make
file was written for linux.


## Quick Start

You can wrap an existing mxArray* with


```cpp
#include "mexmat.h"
  void mexFunction(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
  {
    const mex::Mat<double>(prhs[0]);
  }
```

Note, it is recommended to add a 'const' qualifier when wrapping a 'const'
pointer. This way the array will not be inadvertently modified.

*NOTE*: there is no way in C++ that I know of to force the wrapped object to be
'const' at compile time.

If the underlying type mxArray does not match the template type of the wrapper,
you will get an error. This ensures you get the right type in mex and you will
not have to deal with segfaults due to wrong types.

This design choice was intentional. The point of mex is to speed up some loops
in Matlab. If Matlab is your primary language for the task, then you must be
more concerned with correctness of implementation than other features.

You can also create your own matrix

 ```cpp
   mex::Mat<float> M(3,5);
 ```


Memory for this class  will be managed internally. If you would like to return
the variable back to Matlab, you will have to 'release' it. E.g.:

```cpp
     plhs[0] = my_mat.release();
```

Otherwise, the array data will be destroyed at the end of the object's life.


## using mex::ClassHandle

This allows you to wrap a C++ class and call it from Matlab. The idea to create
a pointer to the class and lock it in memory so that Matlab will not delete it
upon exit from mexFunction.

You will need to explicitly ask Matlab to dispose of the object once you are
done using it.

See the class example in `test/test_class_mex.cc'

We recommend that you wrap your access to the class with another Matlab class.
This way everything is transparent to the user.

*NOTE* ***IMPORTANT*** the Matlab class than you want to use to access the C++ pointer
must inherit from Matlab's 'handle' base class. This makes the class 'by
reference' instead of 'by value.' If you you have a value class, every
instantiation of the class will create a new C++ handle without deleting the
old one. Typically, when forgetting the handle part you will get segfaults and
undesirable side effects that are not easy to diagnose or debug.


## Working with other libraries

### Eigen
If you have Eigen installed you can compile with `-DMEXMAT_WITH_EIGEN`.
This will enable conversion to/from Eigen types. For example:

```cpp
Eigen::Matrix<double, 3, 3> M_eigen;
mex::Mat<double> M(M_eigen); // will copy the data for you

// retuns a map (no copy)
auto M_eigen_map = M.toEigenMap();

// for fixed sized matrices, the ones you know their dimension at run time,
// you can do:
auto M_eigen_map = M.toEigenFixed<3,3>();
```

### OpenCV

Compile with `-DMEXMAT_WITH_OPENCV`. This currently supports opencv2.4. Code has
been tested with single channel images. Relevant code can be found in
`mexmat-cv.h`, which will be included automatically from `mexmat.h`

For example:

```cpp
//
// get an image from Matlab and convert it to an opencv Mat
//
const mex::Mat<uint8_t> I_mex(prhs[0]);
cv::Mat I_cv = mex::mex2cv(I_mex); // this will copy the image

//
// after some opencv processing
//
plhs[0] = mex::cv2mexarray(I_cv);
// or
// plhs[0] = mex::cv2mex(I_cv).release();
```

