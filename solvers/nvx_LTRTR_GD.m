function [X_hat, A, B, info] = nvx_LTRTR_GD(y, op, dim, r, Phi, eta, maxIter, tol)
%% nvx_LTRTR_GD: non-convex low-tubal-rank tensor recovery via factor GD
% Model:
%   min_{A,B}  0.5 * || y - M(A *_Phi B^H) ||_2^2  + 0.125 * || A^H *_Phi A - B^H *_Phi B ||_F^2
%  
% Here M(.) is a linear measurement operator implemented as:
%   y = op.forward(X)
%   M^*(res) = op.adjoint(res)
%
% Inputs:
%   y      : [m,1] measurements
%   op     : struct with fields:
%              op.forward: handle, y = op.forward(X)
%              op.adjoint: handle, X = op.adjoint(y)
%   dim    : [n1,n2,n3]
%   r      : tubal rank
%   Phi    : [n3,n3] unitary/invertible transform
%   eta    : step size (default 1e-3)
%   maxIter: max iterations (default 1000)
%   tol    : stopping tolerance on relative change (default 1e-6)
%
% Outputs:
%   X_hat  : recovered tensor n1 x n2 x n3
%   A,B    : factors (A: n1 x r x n3, B: n2 x r x n3)
%   info   : diagnostics (relRes, relChg, iter)
%
% Requirements:
%   - mode3_product.m
%   - transformed_t_product.m
%   - transformed_tensor_ctranspose.m
%   - transformed_tsvd_skinny.m 

%% -------------------- params & checks --------------------
n1 = dim(1); n2 = dim(2); n3 = dim(3);

if nargin < 6 || isempty(eta);     eta = 1e-3;     end
if nargin < 7 || isempty(maxIter); maxIter = 1000; end
if nargin < 8 || isempty(tol);     tol = 1e-6;     end

y = y(:);
y_norm = max(1, norm(y));

if ~isstruct(op) || ~isfield(op,'forward') || ~isfield(op,'adjoint')
    error('op must be a struct with fields forward and adjoint.');
end
if ~isequal(size(Phi), [n3 n3])
    error('Phi must be n3-by-n3.');
end
if r < 1 || r > min(n1,n2)
    error('Invalid r: must satisfy 1 <= r <= min(n1,n2).');
end

%% -------------------- spectral init (operator adjoint backprojection) --------------------
% X0 = M^*(y)
X0 = op.adjoint(y);   % n1 x n2 x n3

% truncated t-SVD init in Phi-algebra
[U0, S0, V0] = transformed_tsvd_skinny(X0, Phi, r);

% A0 = U0 *_Phi sqrt(S0),  B0 = V0 *_Phi sqrt(S0)
S0_sqrt = fdiag_tensor_sqrt(S0, Phi);

A = transformed_t_product(U0, S0_sqrt, Phi);   % n1 x r x n3
B = transformed_t_product(V0, S0_sqrt, Phi);   % n2 x r x n3

%% -------------------- main loop --------------------
if nargout > 3
    info.relRes = zeros(maxIter,1);
    info.relChg = zeros(maxIter,1);
end

relChg = inf;

for iter = 1:maxIter
    At = A;  Bt = B;

    % Xt = A *_Phi B^H
    Xt = transformed_t_product(At, transformed_tensor_ctranspose(Bt, Phi), Phi);

    % residual: res = M(Xt) - y
    res = op.forward(Xt) - y;

    % Rt = M^*(res)
    Rt = op.adjoint(res);

    % balance term: Qt = A^H *_Phi A - B^H *_Phi B  (r x r x n3)
    Qt = transformed_t_product(transformed_tensor_ctranspose(At, Phi), At, Phi) ...
       - transformed_t_product(transformed_tensor_ctranspose(Bt, Phi), Bt, Phi);

    % gradients
    GradA = transformed_t_product(Rt, Bt, Phi) ...
          + 0.5 * transformed_t_product(At, Qt, Phi);

    GradB = transformed_t_product(transformed_tensor_ctranspose(Rt, Phi), At, Phi) ...
          - 0.5 * transformed_t_product(Bt, Qt, Phi);

    % GD update
    A = At - eta * GradA;
    B = Bt - eta * GradB;

    % stopping criteria
    relChgA = norm(A(:)-At(:)) / max(1, norm(At(:)));
    relChgB = norm(B(:)-Bt(:)) / max(1, norm(Bt(:)));
    relChg  = max(relChgA, relChgB);

    if nargout > 3
        info.relRes(iter) = norm(res) / y_norm;
        info.relChg(iter) = relChg;
    end

    if iter == 1 || mod(iter,100)==0
        fprintf('iter %4d | relChg %.3e | relRes %.3e\n', iter, relChg, norm(res)/y_norm);
    end

    if relChg < tol
        break;
    end
end

X_hat = transformed_t_product(A, transformed_tensor_ctranspose(B, Phi), Phi);

if nargout > 3
    info.iter   = iter;
    info.relRes = info.relRes(1:iter);
    info.relChg = info.relChg(1:iter);
end

fprintf('Stop at iter %d | relChg %.3e\n', iter, relChg);

end
