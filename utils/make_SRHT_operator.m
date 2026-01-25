function op = make_SRHT_operator(dim, m, seed)
%MAKE_SRHT_OPERATOR  Dimension-agnostic SRHT sensing operator without forming dense M.
%
% Usage:
%   op = make_srht_operator(dim, m)
%   op = make_srht_operator(dim, m, seed)
%
% Inputs:
%   dim  : size vector, e.g., [n1,n2] for matrix, or [n1,n2,n3,...,nd] for tensor
%   m    : number of measurements
%   seed : RNG seed (optional, default 0)
%
% Model (on vectorized x = vec(X) of length N = prod(dim)):
%   Np = 2^nextpow2(N)  (zero-padding to power of 2)
%   M  = alpha * P * H * D,  alpha = sqrt(Np/m)
%   D : random +/-1 diagonal (length Np)
%   H : orthonormal Hadamard (FWHT), H'*H = I
%   P : subsampling operator selecting m coordinates
%
% Interfaces:
%   y  = op.forward(X)          size(y) = [m,1]
%   X0 = op.adjoint(y)          size(X0) = dim
%
% Notes:
% - Works for real/complex data (FWHT is linear over C).
% - Requires m <= Np.
% - Complexity: O(Np log Np) per forward/adjoint, no dense matrix.

if nargin < 3 || isempty(seed), seed = 0; end

dim = dim(:).';  % row vector
if isempty(dim) || any(dim <= 0) || any(~isfinite(dim)) || any(mod(dim,1)~=0)
    error('dim must be a vector of positive integers, e.g., [n1,n2,...,nd].');
end

N = prod(dim);
if ~isscalar(m) || m <= 0 || ~isfinite(m) || mod(m,1)~=0
    error('m must be a positive integer.');
end

Np = 2^nextpow2(N);
if m > Np
    error('m must satisfy m <= Np = 2^nextpow2(N). Here N=%d, Np=%d.', N, Np);
end

rng(seed);

% random sign diagonal D (Rademacher)
d = 2*(rand(Np,1) > 0.5) - 1;          % +/- 1 (real)

% subsampling indices Omega (choose m distinct coordinates)
Omega = randperm(Np, m).';             % m x 1

% scaling
alpha = sqrt(Np / m);

% pack operator
op.dim   = dim;
op.N     = N;
op.Np    = Np;
op.m     = m;
op.d     = d;
op.Omega = Omega;
op.alpha = alpha;

op.forward = @(X) srht_forward_general(X, dim, N, Np, d, Omega, alpha);
op.adjoint = @(y) srht_adjoint_general(y, dim, N, Np, d, Omega, alpha);

end

% ===================== forward / adjoint (general dim) =====================

function y = srht_forward_general(X, dim, N, Np, d, Omega, alpha)
% check size
if ~isequal(size(X), dim)
    error('op.forward: input size mismatch. Expected size %s, got %s.', mat2str(dim), mat2str(size(X)));
end

x = X(:);  % vec

% pad
xpad = zeros(Np, 1, 'like', x);
xpad(1:N) = x;

% D
xpad = xpad .* d;

% H (orthonormal)
z = fwht_orth(xpad);

% P + scaling
y = alpha * z(Omega);
end

function X = srht_adjoint_general(y, dim, N, Np, d, Omega, alpha)
y = y(:);
if numel(y) ~= numel(Omega)
    error('op.adjoint: y must be m-by-1 where m=%d.', numel(Omega));
end

% P^T y
z = zeros(Np, 1, 'like', y);
z(Omega) = y;

% H^T z = H z, then scaling alpha
xpad = alpha * fwht_orth(z);

% D^T = D
xpad = xpad .* d;

% unpad and reshape
x = xpad(1:N);
X = reshape(x, dim);
end

% ======================= normalized FWHT =======================
function y = fwht_orth(x)
%FWHT_ORTH  Orthonormal Walsh-Hadamard transform for length N=2^k.
% y = H*x, where H'*H = I.

N = numel(x);
if bitand(N, N-1) ~= 0
    error('FWHT requires length to be power of 2. Got N=%d.', N);
end

y = x;

h = 1;
while h < N
    step = 2*h;
    for i = 1:step:N
        a = y(i:i+h-1);
        b = y(i+h:i+step-1);
        y(i:i+h-1)      = a + b;
        y(i+h:i+step-1) = a - b;
    end
    h = step;
end

y = y / sqrt(N);
end
