function X = transformed_t_product(A, B, Phi)
%TRANSFORMED_T_PRODUCT  Transformed t-product: X = A *_Phi B
% A  : n1 x m  x n3
% B  : m  x n2 x n3
% Phi: n3 x n3 (unitary) transform
% X  : n1 x n2 x n3

assert(ndims(A) == 3 && ndims(B) == 3, 'A and B must be 3rd-order tensors.');
[n1, mA, n3A] = size(A);
[mB, n2, n3B] = size(B);

assert(mA == mB, 'Dimension mismatch: size(A,2) must equal size(B,1).');
assert(n3A == n3B, 'Dimension mismatch: size(A,3) must equal size(B,3).');
assert(isequal(size(Phi), [n3A, n3A]), 'Phi must be n3-by-n3.');

% Transform along mode-3
A_Phi = mode3_product(A, Phi);
B_Phi = mode3_product(B, Phi);

% Slice-wise matrix multiplication
if exist('pagemtimes','file') == 2
    X_Phi = pagemtimes(A_Phi, B_Phi);
else
    X_Phi = zeros(n1, n2, n3A, 'like', A_Phi);
    for k = 1:n3A
        X_Phi(:,:,k) = A_Phi(:,:,k) * B_Phi(:,:,k);
    end
end

% Inverse transform along mode-3 (Phi^H)
X = mode3_product(X_Phi, Phi');
end
