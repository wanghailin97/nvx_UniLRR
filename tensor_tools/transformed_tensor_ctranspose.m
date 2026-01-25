function Xh = transformed_tensor_ctranspose(X, Phi)
%TRANSFORMED_TENSOR_CTRANSPOSE  Tensor conjugate transpose under transform Phi
%
% X  : n1 x n2 x n3 tensor
% Phi: n3 x n3 unitary transform matrix
% Xh : n2 x n1 x n3 tensor

assert(ndims(X) == 3, 'X must be a 3rd-order tensor.');
[n1, n2, n3] = size(X);
assert(isequal(size(Phi), [n3 n3]), 'Phi must be n3-by-n3.');

% Transform
X_Phi = mode3_product(X, Phi);

% Slice-wise conjugate transpose
Xh_Phi = zeros(n2, n1, n3, 'like', X_Phi);
for k = 1:n3
    Xh_Phi(:,:,k) = X_Phi(:,:,k)';   % conjugate transpose
end

% Inverse transform (Phi^H)
Xh = mode3_product(Xh_Phi, Phi');
end
