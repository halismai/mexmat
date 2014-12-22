clear all;
%initialize frequently used matrices and data structs here
[X,map] = imread('RGB_color.gif');
if ~isempty(map)
	Im = ind2rgb(X,map); %generate an RGB image Im
end
int_matrix=int32([1 2 3;4 5 6;7 8 9]);
int64_matrix=int64([1 2 3;4 5 6;7 8 9]);
col_int_matrix=int32([1;2;3]);
double_matrix=100000*rand(3);
col_double_matrix=100000*rand(3,1);
single_val=[1];
complex_matrix=[2-3i 1+i;2-2i 7+i; 9+i 2-4i];
singlep_matrix=single([1.132 2.243 3.232;4.176 5.278 6]);
cell_array={rand(2),rand(3),single(rand(5))};
myCell = {1, 2, 3;'text', rand(5,10,2), {11; 22; 33}};
vect=rand(3,1);
c = TestClass;
%declare a test structure array
patient.name = 'John Doe';
patient.billing = 127.00;
patient.test = [79, 75, 73; 180, 178, 177.5; 220, 210, 205];

%profile -memory on;
%Parameterized test function calls begin here

test_mex('test_getdata',int64_matrix)
test_mex('test_getdata',double_matrix)
test_mex('test_getdata',single_val)
test_mex('test_getdata',[])

test_mex('test_malloc_char', 4)
test_mex('test_malloc_int', 10)
test_mex('test_malloc_double', 5)

test_mex('test_free_int')
test_mex('test_free_char')
test_mex('test_free_double')

test_mex('test_destroyArray')

[out1 out2 out3]=test_mex('test_nargchk_0to3arg', 1, 2)
[out]=test_mex('test_nargchk_1arg')
[out1 out2 out3 out4 out5]=test_mex('test_nargchk_5arg',1, 2, 3, 4)

out=test_mex('test_copy_ctor',double_matrix)
out=test_mex('test_copy_ctor',single_val)
out=test_mex('test_copy_ctor',[])

out=test_mex('test_move_ctor',double_matrix)
out=test_mex('test_move_ctor',single_val)
out=test_mex('test_move_ctor',[])

out=test_mex('test_eigen_double',double_matrix,col_double_matrix)
out=test_mex('test_eigen_int',int_matrix,col_int_matrix)

out=test_mex('test_rand_int',2,3)
out=test_mex('test_rand_float',3,4)
out=test_mex('test_rand_double',4,5)

out=test_mex('test_rand3d_float',3,4,5) %3D INITIALIZATION OPERATOR NOT DEFINED i.e. mat(dim1, dim2, dim3) = value;
out=test_mex('test_rand3d_double',4,5,6) %3D INITIALIZATION OPERATOR NOT DEFINED i.e. mat(dim1, dim2, dim3) = value;

out=test_mex('test_init_RGBimage',Im);

out=test_mex('test_ndims_int',4) %nD INITIALIZATION OPERATOR NOT DEFINED i.e. mat(dim1, dim2, ...) = value;
out=test_mex('test_ndims_float',5) %nD INITIALIZATION OPERATOR NOT DEFINED i.e. mat(dim1, dim2, ...) = value;
out=test_mex('test_ndims_double',6) %nD INITIALIZATION OPERATOR NOT DEFINED i.e. mat(dim1, dim2, ...) = value;

out=test_mex('test_col_int',col_int_matrix) %int64 DOES NOT WORK. int32 WORKS
out=test_mex('test_col_double',col_double_matrix)

test_mex('test_printf','hello world', 34, 34.33) %PROBLEMS WITH OUTPUT, SEE mexmat/test/test_script_output.txt
test_mex('test_printf','just a string', 0, 0.0) %PROBLEMS WITH OUTPUT, SEE mexmat/test/test_script_output.txt

test_mex('test_mat_element_no',int_matrix)
test_mex('test_mat_element_no',double_matrix)
test_mex('test_mat_element_no',single_val)
test_mex('test_mat_element_no',[])

test_mex('test_mat_row_no',int_matrix)
test_mex('test_mat_row_no',double_matrix)
test_mex('test_mat_row_no',single_val)
test_mex('test_mat_row_no',[])

test_mex('test_mat_col_no',int_matrix)
test_mex('test_mat_col_no',double_matrix)
test_mex('test_mat_col_no',single_val)
test_mex('test_mat_col_no',[])

test_mex('test_mat_dim_no',int_matrix)
test_mex('test_mat_dim_no',double_matrix)
test_mex('test_mat_dim_no',single_val)
test_mex('test_mat_dim_no',[])

test_mex('test_mat_dims',int_matrix)
test_mex('test_mat_dims',double_matrix)
test_mex('test_mat_dims',single_val)
test_mex('test_mat_dims',[])

test_mex('test_mat_dims_params')

test_mex('test_isReal',int_matrix)
test_mex('test_isReal',double_matrix)
test_mex('test_isReal',single_val)
test_mex('test_isReal',[])
test_mex('test_isReal','just a string') %STILL DECLARES INPUT AS REAL
test_mex('test_isReal',true) %STILL DECLARES INPUT AS REAL


test_mex('test_isComplex',int_matrix)
test_mex('test_isComplex',double_matrix)
test_mex('test_isComplex',single_val)
test_mex('test_isComplex',[]) 
test_mex('test_isComplex','just a string') 
test_mex('test_isComplex',true) 
test_mex('test_isComplex',complex_matrix) 

test_mex('test_isLogical',double_matrix)
test_mex('test_isLogical',single_val)
test_mex('test_isLogical',[]) 
test_mex('test_isLogical','just a string') 
test_mex('test_isLogical',true)
test_mex('test_isLogical',false) 
test_mex('test_isLogical',(2>1) && (1==1)) 

test_mex('test_isChar',double_matrix)
test_mex('test_isChar',single_val)
test_mex('test_isChar',[]) 
test_mex('test_isChar',complex_matrix)
test_mex('test_isChar',true) 
test_mex('test_isChar','A')
test_mex('test_isChar','just a string') 

test_mex('test_isSingle',double_matrix)
test_mex('test_isSingle',single_val)
test_mex('test_isSingle',[]) 
test_mex('test_isSingle',complex_matrix)
test_mex('test_isSingle',true) 
test_mex('test_isSingle','just a string') 
test_mex('test_isSingle',singlep_matrix) 

test_mex('test_isDouble',complex_matrix)
test_mex('test_isDouble',true) 
test_mex('test_isDouble','just a string') 
test_mex('test_isDouble',singlep_matrix) 
test_mex('test_isDouble',double_matrix)
test_mex('test_isDouble',single_val)
test_mex('test_isDouble',[]) 


test_mex('test_isNumeric',true) 
test_mex('test_isNumeric','just a string')
test_mex('test_isNumeric',complex_matrix)
test_mex('test_isNumeric',singlep_matrix)
test_mex('test_isNumeric',int_matrix) 
test_mex('test_isNumeric',double_matrix)
test_mex('test_isNumeric',single_val)
test_mex('test_isNumeric',[]) 
 
test_mex('test_isNumber',true) 
test_mex('test_isNumber','just a string')
test_mex('test_isNumber',complex_matrix)
test_mex('test_isNumber',singlep_matrix)
test_mex('test_isNumber',int_matrix) 
test_mex('test_isNumber',double_matrix)
test_mex('test_isNumber',single_val)
test_mex('test_isNumber',[]) 

test_mex('test_isCell',true) 
test_mex('test_isCell','just a string')
test_mex('test_isCell',complex_matrix)
test_mex('test_isCell',double_matrix)
test_mex('test_isCell',single_val)
test_mex('test_isCell',[]) 
test_mex('test_isCell',myCell{2}) 
test_mex('test_isCell',myCell) 

test_mex('test_isScalar',true) %LOGICAL EXPRESSION IS TERMED AS A SCALAR 
test_mex('test_isScalar',(2>1) && (1==1)) %LOGICAL EXPRESSION IS TERMED AS A SCALAR  
test_mex('test_isScalar','just a string')
test_mex('test_isScalar',complex_matrix)
test_mex('test_isScalar',double_matrix)
test_mex('test_isScalar',cell_array) 
test_mex('test_isScalar',[])
test_mex('test_isScalar',single_val)
 
test_mex('test_isVector',true) 
test_mex('test_isVector',(2>1) && (1==1)) 
test_mex('test_isVector','just a string') %STRING EXPRESSION IS TERMED AS A VECTOR  
test_mex('test_isVector',complex_matrix)
test_mex('test_isVector',double_matrix)
test_mex('test_isVector',cell_array) %CELL ARRAY HAVING 2D MATRICES IS TERMED AS A VECTOR 
test_mex('test_isVector',[])
test_mex('test_isVector',single_val)
test_mex('test_isVector',vect)

%I do not know how mxIsOpaque evaluates maxArray but for our purposes no input draws a positive response. mxIsOpaque is not documented for Matlab R2014.
test_mex('test_isOpaque',true) 
test_mex('test_isOpaque','just a string') 
test_mex('test_isOpaque',double_matrix)
test_mex('test_isOpaque',cell_array)
test_mex('test_isOpaque',single_val)
test_mex('test_isOpaque',vect)
test_mex('test_isOpaque',patient)

test_mex('test_isStruct',true) 
test_mex('test_isStruct','just a string') 
test_mex('test_isStruct',[])
test_mex('test_isStruct',double_matrix)
test_mex('test_isStruct',cell_array)
test_mex('test_isStruct',single_val)
test_mex('test_isStruct',vect)
test_mex('test_isStruct',patient) 

test_mex('test_isFloatingPoint',true) 
test_mex('test_isFloatingPoint','just a string') 
test_mex('test_isFloatingPoint',patient)
test_mex('test_isFloatingPoint',cell_array)
test_mex('test_isFloatingPoint',int_matrix)
test_mex('test_isFloatingPoint',[])
test_mex('test_isFloatingPoint',vect)
test_mex('test_isFloatingPoint',double_matrix) 

test_mex('test_isClass',true,'logical') 
test_mex('test_isClass','hello world','char') 
test_mex('test_isClass',patient,'struct')
test_mex('test_isClass',cell_array,'cell')
test_mex('test_isClass',[],'double')
test_mex('test_isClass',vect,'double')
test_mex('test_isClass',double_matrix,'double') 
test_mex('test_isClass',int64_matrix,'int64')
test_mex('test_isClass',c,'TestClass')

% Issue for function handle resolved
test_mex('test_isFnHandle',true) 
test_mex('test_isFnHandle','hello world') 
test_mex('test_isFnHandle',c)
test_mex('test_isFnHandle',@test_func)

out=test_mex('test_cell_init',double_matrix, col_double_matrix, vect)
out_other=test_mex('test_cell_init',double_matrix, int_matrix, single_val)
out_other2=test_mex('test_cell_init_mov')

%profile viewer
%p = profile('info');
%profsave(p,'profile_results')
