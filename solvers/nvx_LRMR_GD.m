function [X_hat, A, B, info] = nvx_LRMR_GD(y, op, dim, r, eta, maxIter, tol)
%% nvx_LRMR_GD: non-convex low-rank matrix recovery via factorized GD
%  Model:
%     min_{A,B} 0.5*||y - M(A*B')||_2^2 + 0.125*||A'*A - B'*B||_F^2
%  
% Here M(.) is a linear measurement operator implemented as:
%   y = op.forward(X)
%   M^*(res) = op.adjoint(res)
%
% Inputs:
%   y      : [m,1] measurements
%   op     : measurement operator struct with:
%              op.forward(X) -> y  (m x 1)
%              op.adjoint(y) -> X  (n1 x n2), adjoint of forward
%   dim    : [n1, n2]
%   r      : rank
%   eta    : stepsize (default 1e-3)
%   maxIter: max iterations (default 1000)
%   tol    : tolerance (default 1e-6)
%
% Outputs:
%   X_hat  : recovered matrix
%   A,B    : factors
%   info   : diagnostics (optional)
%
% written by Hailin Wang (wanghailin97@163.com), 2026/01/21

%% -------------------- params & checks --------------------
n1 = dim(1); n2 = dim(2);

if nargin < 5 || isempty(eta);     eta = 1e-3;     end
if nargin < 6 || isempty(maxIter); maxIter = 1000; end
if nargin < 7 || isempty(tol);     tol = 1e-6;     end

y = y(:);
m = numel(y);

if ~isstruct(op) || ~isfield(op,'forward') || ~isfield(op,'adjoint')
    error('op must be a struct with fields forward and adjoint.');
end
if r < 1 || r > min(n1,n2)
    error('Invalid r: must satisfy 1 <= r <= min(n1,n2).');
end

% quick sanity check (optional, can comment out for speed):
% Xtmp = randn(n1,n2);
% ytmp = op.forward(Xtmp);
% if numel(ytmp) ~= m, error('op.forward(X) must output m-by-1 vector.'); end

y_norm = max(1, norm(y));   % for relative residual

%% -------------------- spectral init --------------------
X0 = op.adjoint(y);     % adjoint backprojection, size n1 x n2
if ~isequal(size(X0), [n1,n2])
    error('op.adjoint(y) must return a matrix of size [%d,%d].', n1, n2);
end

[U,S,V] = svd(X0, 'econ');
U0 = U(:,1:r); S0 = S(1:r,1:r); V0 = V(:,1:r);
S0 = sqrt(max(S0,0));
A  = U0*S0;
B  = V0*S0;

%% -------------------- main loop --------------------
if nargout > 3
    info.relRes  = zeros(maxIter,1);
    info.relChg  = zeros(maxIter,1);
end

relChg = inf;

for iter = 1:maxIter
    At = A;  Bt = B;
    Xt = At*Bt';

    % residual: res = M(Xt) - y
    res = op.forward(Xt) - y;

    % Rt = M^*(res)
    Rt = op.adjoint(res);

    % balance term
    Qt = At'*At - Bt'*Bt;

    % gradient terms
    GradA = Rt*Bt  + 0.5*At*Qt;
    GradB = Rt'*At - 0.5*Bt*Qt;

    % GD update
    A = At - eta*GradA;
    B = Bt - eta*GradB;

    % stopping
    relChgA = norm(A-At,'fro') / max(1, norm(At,'fro'));
    relChgB = norm(B-Bt,'fro') / max(1, norm(Bt,'fro'));
    relChg  = max(relChgA, relChgB);

    if nargout > 3
        info.relRes(iter) = norm(res)/y_norm;
        info.relChg(iter) = relChg;
    end

    if iter == 1 || mod(iter,100)==0
        fprintf('iter %4d | relChg %.3e | relRes %.3e\n', ...
            iter, relChg, norm(res)/y_norm);
    end

    if relChg < tol
        break;
    end
end

X_hat = A*B';

if nargout > 3
    info.iter   = iter;
    info.relRes = info.relRes(1:iter);
    info.relChg = info.relChg(1:iter);
end

fprintf('Stop at iter %d | relChg %.3e\n', iter, relChg);
end
