function op = make_gaussian_operator(dim, m, seed)
%MAKE_GAUSSIAN_OPERATOR  Dense Gaussian sensing operator (operator form).
%
% Usage:
%   op = make_gaussian_operator(dim, m)
%   op = make_gaussian_operator(dim, m, seed)
%
% Model:
%   Let N = prod(dim).
%   M ~ N(0, 1/m)^{m x N}.
%
% Interfaces:
%   y  = op.forward(X)          -> size(y) = [m,1]
%   X0 = op.adjoint(y)          -> size(X0) = dim
%
% Notes:
% - This is the operator wrapper of the classical dense Gaussian matrix.
% - Memory cost: O(m*N). (use SRHT / sparse operator if N is large)
% - Adjointness: <M(X), u> = <X, M^*(u)> holds exactly (up to FP error).

if nargin < 3 || isempty(seed)
    seed = 0;
end
rng(seed);

dim = dim(:).';
if isempty(dim) || any(dim <= 0) || any(mod(dim,1)~=0)
    error('dim must be a vector of positive integers.');
end

N = prod(dim);
if ~isscalar(m) || m <= 0 || mod(m,1)~=0
    error('m must be a positive integer.');
end

% dense Gaussian sensing matrix
M = randn(m, N) / sqrt(m);

% pack operator
op.dim = dim;
op.N   = N;
op.m   = m;
op.M   = M;

op.forward = @(X) gaussian_forward(X, M, dim);
op.adjoint = @(y) gaussian_adjoint(y, M, dim);

end

% ===================== forward / adjoint =====================

function y = gaussian_forward(X, M, dim)
if ~isequal(size(X), dim)
    error('op.forward: input size mismatch. Expected %s, got %s.', ...
        mat2str(dim), mat2str(size(X)));
end
y = M * X(:);
end

function X = gaussian_adjoint(y, M, dim)
y = y(:);
X = reshape(M' * y, dim);
end
