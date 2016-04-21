#include "mexmat.h"
#include <Eigen/Dense>
#include <Eigen/LU>
#include <random>
#include <chrono>

//
// takes a unit test function tag along with any parameters.
// produces matrices of informational messages as test results

template <typename Scalar>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <typename _T, class _Dist, class _Rng> inline void fill_random(mex::Mat<_T>& M, _Dist& dist, _Rng& rng)
{
  const mwSize N = M.rows() * M.cols();
  for(mwSize i = 0; i < N; ++i)
    M[i] = dist(rng);
}

template <typename _T, class _Dist, class _Rng> inline void fill_random3(mex::Mat<_T>& M, _Dist& dist, _Rng& rng, mwSize length)
{
  const mwSize N = length;
  for(mwSize i = 0; i < N; ++i)
    M[i] = dist(rng);
}

template <typename _T> void make_random(mex::Mat<_T>& M)
{
  std::mt19937 rng(0);
  std::uniform_int_distribution<_T> dist(0, 16);
  fill_random(M, dist, rng);
}

template <> void make_random<double>(mex::Mat<double>& M)
{
  std::mt19937 rng(0);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  fill_random(M, dist, rng);
}

template <> void make_random<float>(mex::Mat<float>& M)
{
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0, 0.0f);

  fill_random(M, dist, rng);
}

template <typename _T> void make_random3(mex::Mat<_T>& M, mwSize length)
{
  std::mt19937 rng(0);
  std::uniform_int_distribution<_T> dist(0, 16);
  fill_random3(M, dist, rng, length);
}

template <> void make_random3<double>(mex::Mat<double>& M, mwSize length)
{
  std::mt19937 rng(0);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  fill_random3(M, dist, rng, length);
}

template <> void make_random3<float>(mex::Mat<float>& M, mwSize length)
{
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-1.0, 0.0f);

  fill_random3(M, dist, rng, length);
}

double get_double(const mxArray *array_ptr)
{
	double   *pr;
  	mwSize result;
	pr = (double *)mxGetData(array_ptr);
	result = mxGetNumberOfElements(array_ptr);
	static const char* USAGE = "rows and columns must be valid scalars";
 	mex::nargchk(1,1, result, USAGE);
	return (*pr);
}

/* Display the subscript associated with the given index. */
void display_subscript(const mxArray *array_ptr, mwSize index)
{
	mwSize     inner, subindex, total, d, q, number_of_dimensions;
	mwSize       *subscript;
	const mwSize *dims;

	number_of_dimensions = mxGetNumberOfDimensions(array_ptr);
	subscript = (mwSize*)mxCalloc(number_of_dimensions, sizeof(mwSize));
	dims = mxGetDimensions(array_ptr);

	mexPrintf("(");
	subindex = index;
	for (d = number_of_dimensions-1; ; d--)
	{ /* loop termination is at the end */
		for (total=1, inner=0; inner<d; inner++)
			total *= dims[inner];

		subscript[d] = subindex / total;
		subindex = subindex % total;

		if (d == 0)
		{
			break;
		}
	}

	for (q=0; q<number_of_dimensions-1; q++)
	{
		mexPrintf("%d,", subscript[q] + 1);
	}
	mexPrintf("%d)", subscript[number_of_dimensions-1] + 1);
	mxFree(subscript);
}

template <typename _T> void test_getdata(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "output = fn('test_getdata,data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	double   *pr;
  	mwSize result, index;
	pr = (_T*)mex::getData(prhs[1]);
	result = mxGetNumberOfElements(prhs[1]);

	for (index=0; index<mwSize(result); index++)
	{
		mexPrintf("\t");
		display_subscript(prhs[1], index);
		mexPrintf(" = %f\n", *pr++);
	}

	mexPrintf("\nData Retreival (getData) test completed.\n");
}

void test_mat_element_no(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_mat_element_no',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
 	mex::nargchk(0,0, nlhs, USAGE);

	mwSize result=mex::numel(prhs[1]);
	mexPrintf("\nTotal number of elements in given array equal %d. Test completed.\n", result);
}

void test_mat_row_no(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_mat_row_no',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
 	mex::nargchk(0,0, nlhs, USAGE);

	mwSize result=mex::rows(prhs[1]);
	mexPrintf("\nTotal number of rows in given matrix equal %d. Test completed.\n", result);
}

void test_mat_col_no(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_mat_col_no',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
 	mex::nargchk(0,0, nlhs, USAGE);

	mwSize result=mex::cols(prhs[1]);
	mexPrintf("\nTotal number of columns in given matrix equal %d. Test completed.\n", result);
}

void test_mat_dim_no(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_mat_dim_no',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	mwSize result=mex::ndims(prhs[1]);
	mexPrintf("\nTotal number of dimensions for given matrix equal %d. Test completed.\n", result);
}

void test_mat_dims(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_mat_dims',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	const mwSize  *dims;
	dims=mex::dims(prhs[1]);
	mwSize result=mex::ndims(prhs[1]);
	mexPrintf("\nTotal number of dimensions for given matrix equal %d.\n", result);
	for(int i=0;i<int(result);i++)
	{
		mexPrintf("Dimension %d has %d elements.\n",i+1,dims[i]);
	}
	mexPrintf("Test completed.\n");
}

void test_mat_dims_params(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	//We prepare and send matrices within c program here since we need to use constant expressions for testing here
	static const char* USAGE = "fn('test_mat_dims_params')";
 	mex::nargchk(1,1, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	const mwSize rows=3;
	const mwSize cols=4;

  	std::mt19937 rng(0); // fixed sequence
  	std::uniform_real_distribution<> dist(-1.0, 1.0);

	mex::Mat<double> temp(rows, cols);

	for(mwSize i=0; i<rows*cols; ++i)
	{
    		temp[i] = dist(rng);
	}


	mwSize result=mex::ndims(temp);
	mexPrintf("\nTotal number of dimensions for given matrix equal %d.\n", result);

	mexPrintf("Dimension %d has %d elements.\n",1,mex::dim<0>(temp));
	mexPrintf("Dimension %d has %d elements.\n",2,mex::dim<1>(temp));

	mexPrintf("Test completed.\n");
}

void test_length(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_length',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	mwSize result=mex::length(prhs[1]);
	mexPrintf("\nLength of given input equals %d. Test completed.\n", result);
}

void test_isReal(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[]) //returns true even if a string is provided
{
	static const char* USAGE = "fn('test_isReal',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	bool result=false;
	result=mex::isReal(prhs[1]);
	if (result == true)
	{
		mexPrintf("\nThe input is in form of real numbers.\n");
	}else
	{
		mexPrintf("\nThe input is not in the form of real numbers.\n");
	}
	mexPrintf("Test completed.\n");
}

void test_isComplex(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_isComplex',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	bool result=false;
	result=mex::isComplex(prhs[1]);
	if (result == true)
	{
		mexPrintf("\nThe input is in form of complex numbers.\n");
	}else
	{
		mexPrintf("\nThe input is not in the form of complex numbers.\n");
	}
	mexPrintf("Test completed.\n");
}

void test_isLogical(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_isLogical',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);
	bool result=false;

	result=mex::isLogical(prhs[1]);
	if (result == true)
	{
		mexPrintf("\nThe input is in form of logical expression.\n");
	}else
	{
		mexPrintf("\nThe input is not in form of logical expression.\n");
	}
	mexPrintf("Test completed.\n");
}

void test_isChar(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_isChar',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	bool result=false;
	result=mex::isChar(prhs[1]);
	if (result == true)
	{
		mexPrintf("\nThe input is in form of characters.\n");
	}else
	{
		mexPrintf("\nThe input is not in form of characters.\n");
	}
	mexPrintf("Test completed.\n");
}

void test_isDouble(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_isDouble',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	bool result=false;
	result=mex::isDouble(prhs[1]);
	if (result == true)
	{
		mexPrintf("\nThe input is in form of double data.\n");
	}else
	{
		mexPrintf("\nThe input is not in form of double data.\n");
	}
	mexPrintf("Test completed.\n");
}

void test_isSingle(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_isSingle',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	bool result=false;
	result=mex::isSingle(prhs[1]);
	if (result == true)
	{
		mexPrintf("\nThe input is in form of single data.\n");
	}else
	{
		mexPrintf("\nThe input is not in form of single data.\n");
	}
	mexPrintf("Test completed.\n");
}

void test_isNumeric(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_isNumeric',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	bool result=false;
	result=mex::isNumeric(prhs[1]);
	if (result == true)
	{
		mexPrintf("\nThe input is in form of numeric data.\n");
	}else
	{
		mexPrintf("\nThe input is not in form of numeric data.\n");
	}
	mexPrintf("Test completed.\n");
}

void test_isNumber(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_isNumber',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	bool result=false;
	result=mex::isNumber(prhs[1]);
	if (result == true)
	{
		mexPrintf("\nThe input is in form of number format.\n");
	}else
	{
		mexPrintf("\nThe input is not in form of number format.\n");
	}
	mexPrintf("Test completed.\n");

}

void test_isCell(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_isCell',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	bool result=false;
	result=mex::isCell(prhs[1]);
	if (result == true)
	{
		mexPrintf("\nThe input is in form of cell format.\n");
	}else
	{
		mexPrintf("\nThe input is not in form of cell format.\n");
	}
	mexPrintf("Test completed.\n");

}

void test_isStruct(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_isStruct',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	bool result=false;
	result=mex::isStruct(prhs[1]);
	if (result == true)
	{
		mexPrintf("\nThe input is in the form of Structure format.\n");
	}else
	{
		mexPrintf("\nThe input is not in form of Struture format.\n");
	}
	mexPrintf("Test completed.\n");

}

void test_isOpaque(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_isOpaque',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	bool result=false;
	result=mex::isOpaque(prhs[1]);

	if (result == true)
	{
		mexPrintf("\nThe input is in the form of Opaque format.\n");
	}else
	{
		mexPrintf("\nThe input is not in the form of Opaque format.\n");
	}
	mexPrintf("Test completed.\n");

}

void test_isClass(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_isClass',data, 'DATATYPE')";
 	mex::nargchk(3,3, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

 	std::string name=mex::getString( prhs[2] );
	bool result=false;
	result=mex::isClass(prhs[1],name);
	if (result == true)
	{
		mexPrintf("\nThe input is of type %s.\n",name.c_str());
	}else
	{
		mexPrintf("\nThe input is not of type %s.\n",name.c_str());
	}
	mexPrintf("Test completed.\n");

}

void test_isFnHandle(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_isFnHandle',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	bool result=false;
	result=mex::isFnHandle(prhs[1]);
	if (result == true)
	{
		mexPrintf("\nThe input is in form of function handle format.\n");
	}else
	{
		mexPrintf("\nThe input is not in form of function handle format.\n");
	}
	mexPrintf("Test completed.\n");
}

void test_isFloatingPoint(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_isFloatingPoint',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	bool result=false;
	result=mex::isFloatingPoint(prhs[1]);
	if (result == true)
	{
		mexPrintf("\nThe input is in form of floating point data.\n");
	}else
	{
		mexPrintf("\nThe input is not in form of floating point data.\n");
	}
	mexPrintf("Test completed.\n");

}

void test_isVector(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_isVector',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	bool result=false;
	result=mex::isVector(prhs[1]);
	if (result == true)
	{
		mexPrintf("\nThe input is in form of Vector data.\n");
	}else
	{
		mexPrintf("\nThe input is not in form of Vector data.\n");
	}
	mexPrintf("Test completed.\n");

}

void test_isScalar(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_isScalar',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	bool result=false;
	result=mex::isScalar(prhs[1]);
	if (result == true)
	{
		mexPrintf("\nThe input is in form of Scalar data.\n");
	}else
	{
		mexPrintf("\nThe input is not in form of Scalar data.\n");
	}
	mexPrintf("Test completed.\n");

}

template <typename _T> void test_copy_ctor(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
 	static const char* USAGE = "output = fn('test_copy_ctor',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
 	mex::nargchk(1,1, nlhs, USAGE);

 	const mex::Mat<_T> input(prhs[1]);
	// return a copy of input
  	mex::Mat<_T> output(input);
  	plhs[0] = output.release();
	mexPrintf("\nCopy constructor test completed. Copied data is returned.\n");
}

template <typename _T> void test_move_ctor(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
 	static const char* USAGE = "output = fn('test_move_ctor',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
 	mex::nargchk(1,1, nlhs, USAGE);

 	mex::Mat<_T> input(prhs[1]);
	// return a copy of input
  	mex::Mat<_T> output(std::move(input));
  	plhs[0] = output.release();
	mexPrintf("\nMove constructor test completed. Moved data is returned.\n");
}


template <typename _T> void test_eigen_lin_solver(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
  	static const char* USAGE = "output = fn('test_eigen_lin_solver',A_matrix,b_vector)";
 	mex::nargchk(3,3, nrhs, USAGE);
 	mex::nargchk(1,1, nlhs, USAGE);

	const mex::Mat<_T> A(prhs[1]);
	const mex::Mat<_T> b(prhs[2]);

	//
	// make sure the we have more equations than unknowns
	//
	mex::massert( A.rows() <= A.cols() );
	mex::massert( A.rows() == b.rows() );

	const auto A_eigen = A.toEigen();
	const auto b_eigen = b.toEigen();

	Matrix<_T> x = A_eigen.fullPivLu().solve(b_eigen);

	mex::Mat<_T> ret(x);
	plhs[0] = ret.release();
	mexPrintf("\nEigen matrix based linear solver test completed. Solution is returned.\n");
}

template <typename _T> void test_ndims_init(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
 	static const char* USAGE = "output = fn('test_dims_init',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
 	mex::nargchk(1,1, nlhs, USAGE);

 	mwSize ndims = static_cast<mwSize>(get_double(prhs[1]) );
 	mwSize *dims = new mwSize[ndims];

 	for(mwSize i=0; i < ndims; ++i)
 	{
 		dims[i] = (rand()%9) + 1;
 	}

 	mex::Mat<_T> input(ndims,*dims);

	plhs[0]=input.release();
	mexPrintf("\nMulti-dimensional initialization operator for nD matrix is not defined though its defined for 2D matrix access i.e. Mat(i,j).\nnD matrix declaration test completed.");
}

void test_destroyArray(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
 	static const char* USAGE = "output = fn('test_destroyArray')";
 	mex::nargchk(1,1, nrhs, USAGE);
 	mex::nargchk(0,0, nlhs, USAGE);

 	mwSize rows=3;
	mwSize cols=4;
	// expecting a _T class
	mex::Mat<double> temp(rows,cols);

	make_random(temp);

	mexPrintf("3 x 4 Matrix initialized.\n");
	mxArray* array_ref;
	array_ref=temp.release();
	mwSize nelem = mxGetNumberOfElements(array_ref);
 	mex::destroyArray(array_ref);
 	mexPrintf("\nMatrix array destroyed. Number of matrix elements destroyed %d. Test completed.\n", nelem);
}

template <typename _T> void test_col_init(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
 	static const char* USAGE = "output = fn('test_col_init',data)";
 	mex::nargchk(2,2, nrhs, USAGE);
 	mex::nargchk(1,1, nlhs, USAGE);

 	if (mex::isVector(prhs[1]) )
 	{
		mex::Mat<_T> input(prhs[1]);
 		std::vector<_T> vect(input.rows());
 		for(mwSize i=0; i<input.rows(); ++i)
		{
			for(mwSize j=0; j<input.cols(); ++j)
			{
				vect.at(i)=input(i,j);
			}
		}
		// return a copy of input
	  	mex::Mat<_T> output(vect);
 	 	plhs[0] = output.release();
		mexPrintf("\nVector init constructor test completed. Vector data is returned.\n");
	}else
	{
		mexPrintf("\nVector init constructor test not completed. Please provide a column vector.\n");
	}
}

void test_cell_init(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "output = fn('test_cell_init',double matrix,double matrix,double matrix)";
 	mex::nargchk(4,4, nrhs, USAGE);
	mex::nargchk(1,1, nlhs, USAGE);

	// via the class warpper
	mex::Cell C((mwSize)(nrhs-1),1);

	C.set( 0, mxDuplicateArray(prhs[1]) );
        C.set( 1, mxDuplicateArray(prhs[2]) );
        C.set( 2, mxDuplicateArray(prhs[3]) );

	plhs[0]=C.release();

	mexPrintf("\nCell initilalization test completed.\n");
	//mxFree(C);
}

void test_cell_init_mov(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "output = fn('test_cell_init_mov')";
 	mex::nargchk(1,1, nrhs, USAGE);
	mex::nargchk(1,1, nlhs, USAGE);

	// via the class warpper
	mwSize rows=3, cols=4;
	mex::Cell C(rows,cols);

	for (mwSize i=0;i<rows*cols;i++)
	{
		C.set(i,mex::newMexMatrix(2,2));
	}

	mex::Cell D(std::move(C));

	plhs[0]=D.release();

	mexPrintf("\nCell initilalization using move-constructor test completed.\n");
}


template <typename _T>
void test_rand_init(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	//mexPrintf("\n%d\n",nrhs);
	static const char* USAGE = "output = fn('test_rand_DATATYPE',rows,cols)";
 	mex::nargchk(3,3, nrhs, USAGE);
	mex::nargchk(1,1, nlhs, USAGE);

	mwSize rows = static_cast<mwSize>( get_double(prhs[1]) );
	mwSize cols = static_cast<mwSize>( get_double(prhs[2]) );

	mex::Mat<_T> temp(rows,cols);
    	make_random(temp);
	plhs[0]=temp.release();


	mexPrintf("\nRandom matrix generation test completed. Number of bytes used by data type are %d.\n",sizeof(_T));
}

template <typename _T> void test_init3d_ctor(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	//mexPrintf("\n%d\n",nrhs);
	static const char* USAGE = "output = fn('test_rand3d_DATATYPE',dim#1,dim#2,dim#3)";
 	mex::nargchk(4,4, nrhs, USAGE);
	mex::nargchk(1,1, nlhs, USAGE);

	mwSize dim1=get_double(prhs[1]);
	mwSize dim2=get_double(prhs[2]);
	mwSize dim3=get_double(prhs[3]);

	// expecting a _T class
	mex::Mat<_T> temp(dim1,dim2,dim3);

	for(mwSize i=0; i<dim1; i++)
	{
		for(mwSize j=0; j<dim2; j++)
		{
			for(mwSize k=0; k<dim3; k++)
			{
				temp(i,j,k)=(rand()%100)/100.0;
			}
		}
	}

	plhs[0]=temp.release();
	mexPrintf("\nMat(i,j,k) initialization operator used.\n3D matrix declaration test completed.");
}

void test_init_RGBimage(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	//mexPrintf("\n%d\n",nrhs);
	static const char* USAGE = "output = fn('test_init_RGBimage',RGB_Image)";
 	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(1,1, nlhs, USAGE);

	mex::Mat<double> input(prhs[1]);
	mex::Mat<double> Img(input.rows(),input.cols(),input.depth());

	for(mwSize i=0; i<input.rows(); i++)
	{
		for(mwSize j=0; j<input.cols(); j++)
		{
			for(mwSize k=0; k<input.depth(); k++)
			{
				Img(i,j,k)=input(i,j,k);
			}
		}
	}

	plhs[0]=Img.release();
	mexPrintf("\nRGB Image based matrix initialization complete.\n3D matrix based Image initialization test completed.");
}

void test_error_msg(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_error_msg')";
 	mex::nargchk(1,1, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);
  	mex::error("Error message function successfully tested");
}

void test_warning_msg(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_warning_msg')";
 	mex::nargchk(1,1, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);
  	mex::warning("Warning message function successfully tested");
}

void test_printf(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_printf', 'STRING',INTEGER, DOUBLE)";
 	mex::nargchk(4,4, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

    	std::string str2= mex::getString(prhs[1]);
  	int var1=get_double(prhs[2]);
  	double var2=get_double(prhs[3]);
    	printf("var1: %d var2 %f\n", var1, mex::getNumber<double>(prhs[3]));
  	mex::printf("Testing mexmat printf function.\nint var = %d\ndouble var = %f\nstring var = %s\n\n",int(var1),var2,str2.c_str());
  	mexPrintf("Same data displayed through mexPrintF function.\nint var = %d\ndouble var = %f\nstring var = %s\n",int(var1),var2,str2.c_str());
}

void test_massert(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
  	static const char* USAGE = "fn('test_massert')";
	mex::nargchk(1,1, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	int large=10;
	int small=2;
	double equal1= 5.0;
	double equal2= 5.0;
	long int large_l = 100000000000;
	long int small_l = 1000000;
	std::string equalstr1 = "hello";
	std::string equalstr2 = "hello";

	mex::massert((large < small) && (equal1 != equal2) && (large_l < small_l) && (equalstr1 != equalstr2),"Testing massert function. All sub-conditions are false. Test completed.");
}

void test_nargchk_1arg(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	  static const char* USAGE_LHS = "Left hand side arguments either less or more than required number";
	  static const char* USAGE_RHS = "Right hand side arguments either less or more than required number";

	  mex::nargchk(1,1,nlhs,USAGE_LHS);
	  mex::nargchk(1,1,nrhs,USAGE_RHS);

	  mexPrintf("\nArgument check completed\n");

	  mex::Mat<double> temp(1,1);
	  temp(0,0)=0;

	  plhs[0]=temp.release();
}

void test_nargchk_5arg(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	  static const char* USAGE_LHS = "Left hand side arguments either less or more than required number";
	  static const char* USAGE_RHS = "Right hand side arguments either less or more than required number";

	  mex::nargchk(5,5,nlhs,USAGE_LHS);
	  mex::nargchk(5,5,nrhs,USAGE_RHS);
	  mexPrintf("\nArgument check completed\n");

	  mex::Mat<double> temp(1,1);
	  temp(0,0)=0;

	  for(int i=0;i<5;i++)
	  	plhs[i]=temp.release();
}

void test_nargchk_0to3arg(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE_LHS = "Left hand side arguments either less or more than required number";
	static const char* USAGE_RHS = "Right hand side arguments either less or more than required number";

	mex::nargchk(0,3,nlhs,USAGE_LHS);
	mex::nargchk(0,3,nrhs,USAGE_RHS);
	mexPrintf("\nArgument check completed\n");

	mex::Mat<double> temp(1,1);
	temp(0,0)=0;

	plhs[0]=temp.release();

	if (nlhs > 1)
	plhs[1]=temp.release();

	if (nlhs > 2)
	plhs[2]=temp.release();
}

void test_getString(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_getString, 'STRING')";
	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);
	const std::string command = mex::getString( prhs[1] );
        mexPrintf("\nThe input string sent via command line is \"%s\". Test completed\n", command.c_str());
}

template <typename _T> void test_malloc(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_malloc_DATATYPE, SIZEOFBUFFER)";
	mex::nargchk(2,2, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	mwSize size_of_buffer=get_double(prhs[1]);
	_T* buf = mex::malloc<_T>(size_of_buffer);

	if (buf != NULL)
	{
	  	mexPrintf("\n%d bytes allocated. Memory allocation test completed.\n", sizeof(_T)*size_of_buffer);
	  	mex::free<_T>(buf);
	}else
	{
	  	mexPrintf("\nMemory allocation test failed\n");
	}
}

template <typename _T> void test_free(int nlhs, mxArray* plhs[], int nrhs, mxArray const* prhs[])
{
	static const char* USAGE = "fn('test_free_DATATYPE)";
	mex::nargchk(1,1, nrhs, USAGE);
	mex::nargchk(0,0, nlhs, USAGE);

	mwSize size_of_buffer=4;
	_T* buf = mex::malloc<_T>(size_of_buffer);

	if (buf != NULL)
	{
	  	mexPrintf("\n%d bytes allocated.\n", sizeof(_T)*size_of_buffer);
	}else
	{
	  	mexPrintf("\nMemory allocation action failed\n");
	}

  	mex::free<_T>(buf);
	mexPrintf("\nObject freed successfully. Test Completed\n");
}

void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, mxArray const* prhs[])
{
	// first argument is a string of the command to perform
	if(nrhs < 1)
	{
		mex::error("must have at least one argument");
	}

	const std::string command = mex::getString( prhs[0] );
	mexPrintf("\n-------------------------------------------\n");

	if ("test_init_RGBimage" == command)
	{
		test_init_RGBimage(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_cell_init_mov" == command)
	{
		test_cell_init_mov(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_cell_init" == command)
	{
		test_cell_init(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_eigen_double" == command)
	{
		test_eigen_lin_solver<double>(nlhs, plhs, nrhs, prhs);
	}
	else if("test_rand_int" == command)
	{
		test_rand_init<int>(nlhs, plhs, nrhs, prhs);
	}
	else if("test_rand_double" == command)
	{
		test_rand_init<double>(nlhs, plhs, nrhs, prhs);
	}
	else if("test_rand_float" == command)
	{
		test_rand_init<float>(nlhs, plhs, nrhs, prhs);
	}
	else if("test_rand3d_float" == command)
	{
		test_init3d_ctor<float>(nlhs, plhs, nrhs, prhs);
	}
	else if("test_rand3d_int" == command)
	{
		test_init3d_ctor<int>(nlhs, plhs, nrhs, prhs);
	}
	else if("test_rand3d_double" == command)
	{
		test_init3d_ctor<double>(nlhs, plhs, nrhs, prhs);
	}
	else if("test_ndims_double" == command)
	{
		test_ndims_init<double>(nlhs, plhs, nrhs, prhs);
	}
	else if("test_ndims_int" == command)
	{
		test_ndims_init<int>(nlhs, plhs, nrhs, prhs);
	}
	else if("test_ndims_float" == command)
	{
		test_ndims_init<float>(nlhs, plhs, nrhs, prhs);
	}
	else if("test_col_double" == command)
	{
		test_col_init<double>(nlhs, plhs, nrhs, prhs);
	}
	else if("test_col_int" == command)
	{
		test_col_init<int>(nlhs, plhs, nrhs, prhs);
	}

	else if("test_getdata" == command)
	{
		test_getdata<double>(nlhs, plhs, nrhs, prhs);
	}
	else if("test_length" == command)
	{
		test_length(nlhs, plhs, nrhs, prhs);
	}else if("test_copy_ctor" == command)
	{
		test_copy_ctor<double>(nlhs, plhs, nrhs, prhs);
	}
	else if("test_move_ctor" == command)
	{
		test_move_ctor<double>(nlhs, plhs, nrhs, prhs);
	}
	else if("test_error_msg" == command)
	{
		test_error_msg(nlhs, plhs, nrhs, prhs);
	}
	else if("test_warning_msg" == command)
	{
		test_warning_msg(nlhs, plhs, nrhs, prhs);
	}
	else if("test_printf" == command)
	{
		test_printf(nlhs, plhs, nrhs, prhs);
	}
	else if("test_massert" == command)
	{
		test_massert(nlhs, plhs, nrhs, prhs);
	}
	else if("test_nargchk_1arg" == command)
	{
		test_nargchk_1arg(nlhs, plhs, nrhs, prhs);
	}
	else if("test_nargchk_5arg" == command)
	{
		test_nargchk_5arg(nlhs, plhs, nrhs, prhs);
	}
	else if("test_nargchk_0to3arg" == command)
	{
		test_nargchk_0to3arg(nlhs, plhs, nrhs, prhs);
	}
	else if("test_getString" == command)
	{
		test_getString(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_malloc_int" == command)
	{
		test_malloc<int>(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_malloc_double" == command)
	{
		test_malloc<double>(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_malloc_char" == command)
	{
		test_malloc<char>(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_destroyArray" == command)
	{
		test_destroyArray(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_free_int" == command)
	{
		test_free<int>(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_free_double" == command)
	{
		test_free<double>(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_free_char" == command)
	{
		test_free<char>(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_mat_element_no" == command)
	{
		test_mat_element_no(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_mat_row_no" == command)
	{
		test_mat_row_no(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_mat_col_no" == command)
	{
		test_mat_col_no(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_mat_dim_no" == command)
	{
		test_mat_dim_no(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_mat_dims" == command)
	{
		test_mat_dims(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_mat_dims_params" == command)
	{
		test_mat_dims_params(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_isReal" == command)
	{
		test_isReal(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_isComplex" == command)
	{
		test_isComplex(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_isLogical" == command)
	{
		test_isLogical(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_isChar" == command)
	{
		test_isChar(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_isSingle" == command)
	{
		test_isSingle(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_isDouble" == command)
	{
		test_isDouble(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_isNumeric" == command)
	{
		test_isNumeric(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_isNumber" == command)
	{
		test_isNumber(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_isCell" == command)
	{
		test_isCell(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_isScalar" == command)
	{
		test_isScalar(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_isVector" == command)
	{
		test_isVector(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_isOpaque" == command)
	{
		test_isOpaque(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_isStruct" == command)
	{
		test_isStruct(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_isFloatingPoint" == command)
	{
		test_isFloatingPoint(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_isClass" == command)
	{
		test_isClass(nlhs, plhs, nrhs, prhs);
	}
	else if ("test_isFnHandle" == command)
	{
		test_isFnHandle(nlhs, plhs, nrhs, prhs);
	}
	else
	{
		std::stringstream msg;
		msg << "unknown command '" << command << "'\n";
		mex::error(msg.str());
	}
}

