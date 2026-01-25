function [U, S, V] = transformed_tsvd(X, Phi, use_econ)
%TRANSFORMED_TSVD  t-SVD under a unitary transform Phi along mode-3
%
% X   : n1 x n2 x n3
% Phi : n3 x n3 unitary transform
% use_econ (optional): true (default) uses svd(...,'econ') and embeds to full.
%
% Output (full-size tensors):
% U : n1 x n1 x n3
% S : n1 x n2 x n3
% V : n2 x n2 x n3

if nargin < 3 || isempty(use_econ)
    use_econ = true;
end

assert(ndims(X) == 3, 'X must be a 3rd-order tensor.');
[n1, n2, n3] = size(X);
assert(isequal(size(Phi), [n3 n3]), 'Phi must be n3-by-n3.');

% Transform
X_Phi = mode3_product(X, Phi);

% Preallocate
U_Phi = zeros(n1, n1, n3, 'like', X_Phi);
S_Phi = zeros(n1, n2, n3, 'like', X_Phi);
V_Phi = zeros(n2, n2, n3, 'like', X_Phi);

if use_econ
    r = min(n1, n2);

    for k = 1:n3
        Xk = X_Phi(:,:,k);
        [Uk, Sk, Vk] = svd(Xk, 'econ');  % Uk: n1xr, Sk: rxr, Vk: n2xr

        % Embed to full
        Ufull = zeros(n1, n1, 'like', X_Phi);
        Vfull = zeros(n2, n2, 'like', X_Phi);
        Sfull = zeros(n1, n2, 'like', X_Phi);

        Ufull(:,1:r)   = Uk;
        Vfull(:,1:r)   = Vk;
        Sfull(1:r,1:r) = Sk;

        U_Phi(:,:,k) = Ufull;
        S_Phi(:,:,k) = Sfull;
        V_Phi(:,:,k) = Vfull;
    end
else
    for k = 1:n3
        [U_Phi(:,:,k), S_Phi(:,:,k), V_Phi(:,:,k)] = svd(X_Phi(:,:,k));
    end
end

% Inverse transform
U = mode3_product(U_Phi, Phi');
S = mode3_product(S_Phi, Phi');
V = mode3_product(V_Phi, Phi');
end
