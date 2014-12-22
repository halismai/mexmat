A = rand(2, 10);
b = rand(2, 1);
x = test_eigen_mex(A, b);


fprintf('difference from matlab %f\n', norm(x - (A\b)));
