function Y = mode3_product(X, M)
%MODE3_PRODUCT  Mode-3 product: Y = X ¡Á_3 M
% X: n1 x n2 x n3 tensor
% M: m  x n3 matrix
% Y: n1 x n2 x m tensor

assert(ndims(X) == 3, 'X must be a 3rd-order tensor.');
[n1, n2, n3] = size(X);

assert(ismatrix(M), 'M must be a 2-D matrix.');
[m, n3_M] = size(M);
assert(n3 == n3_M, 'Dimension mismatch: size(M,2) must equal size(X,3).');

% Unfold along mode-3: (n3) x (n1*n2)
X3 = reshape(permute(X, [3 1 2]), n3, []);

% Multiply: (m x n3) * (n3 x n1*n2) = (m x n1*n2)
Y3 = M * X3;

% Fold back: (n1 x n2 x m)
Y = permute(reshape(Y3, [m, n1, n2]), [2 3 1]);
end
