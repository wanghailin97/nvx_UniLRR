function [U, S, V] = transformed_tsvd_skinny(X, Phi, r)
%TRANSFORMED_TSVD_SKINNY  Skinny (econ) t-SVD under unitary transform Phi
%
%   [U,S,V] = transformed_tsvd_skinny(X, Phi)
%   [U,S,V] = transformed_tsvd_skinny(X, Phi, r)
%
% X   : n1 x n2 x n3 tensor
% Phi : n3 x n3 unitary transform matrix
% r   : (optional) truncation rank, 1 <= r <= min(n1,n2)
%
% Output:
% U : n1 x r x n3
% S : r  x r x n3
% V : n2 x r x n3
%
% In transform domain: X_Phi(:,:,k) = U_Phi(:,:,k) * S_Phi(:,:,k) * V_Phi(:,:,k)^H

assert(ndims(X) == 3, 'X must be a 3rd-order tensor.');
[n1, n2, n3] = size(X);
assert(isequal(size(Phi), [n3 n3]), 'Phi must be n3-by-n3.');

rmax = min(n1, n2);
if nargin < 3 || isempty(r)
    r = rmax;
else
    assert(isscalar(r) && r >= 1 && r <= rmax, 'r must satisfy 1 <= r <= min(n1,n2).');
    r = floor(r);
end

% 1) Transform along mode-3
X_Phi = mode3_product(X, Phi);

% 2) Slice-wise econ SVD (and optional truncation to rank-r)
U_Phi = zeros(n1, r, n3, 'like', X_Phi);
S_Phi = zeros(r,  r, n3, 'like', X_Phi);
V_Phi = zeros(n2, r, n3, 'like', X_Phi);

for k = 1:n3
    [Uk, Sk, Vk] = svd(X_Phi(:,:,k), 'econ');  % Uk: n1 x rmax, Sk: rmax x rmax, Vk: n2 x rmax
    U_Phi(:,:,k) = Uk(:,1:r);
    S_Phi(:,:,k) = Sk(1:r,1:r);
    V_Phi(:,:,k) = Vk(:,1:r);
end

% 3) Inverse transform (Phi^H)
U = mode3_product(U_Phi, Phi');
S = mode3_product(S_Phi, Phi');
V = mode3_product(V_Phi, Phi');
end
