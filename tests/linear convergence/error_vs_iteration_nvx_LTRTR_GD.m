%% test the linear convergence of nvx_LTRTR_GD
% Fro-norm and Phi-spectral-norm based relative error v.s. iteration

clear; close all;
addpath(genpath(pwd));
rng(0);

%% -------------------- simulated random low-tubal-rank tensor --------------------
n = 60;                 % n1=n2=n
n1 = n; n2 = n;
n3 = 5;                % 3rd mode length
dim = [n1,n2,n3];
r  = 3;                 % tubal rank (factor width)

% Construct a low-tubal-rank tensor: X = A *_Phi B^H
Phi = dftmtx(n3)/sqrt(n3);               % unitary transform (DFT)

A0 = randn(n1, r, n3);% + 1i*randn(n1, r, n3);
B0 = randn(n2, r, n3);% + 1i*randn(n2, r, n3);

X  = transformed_t_product(A0, transformed_tensor_ctranspose(B0, Phi), Phi);

%% -------------------- settings --------------------
% measurements
c_m = 5;
m = ceil(c_m*n*r*n3);
op = make_Gaussian_operator(dim, m);

% noiseless measurements (you can add noise if needed)
y = op.forward(X);
y_norm = max(1, norm(y));

eta     = 1e-3;
maxIter = 1000;
tol     = 1e-12;

Error_fro  = zeros(maxIter, 1);
Error_spec = zeros(maxIter, 1);

%% -------------------- spectral init (tensor) --------------------
% backprojection
X0 = op.adjoint(y);   % n1 x n2 x n3

% skinny t-SVD init
[U0, S0, V0] = transformed_tsvd_skinny(X0, Phi, r);
S0_sqrt = fdiag_tensor_sqrt(S0, Phi);

A = transformed_t_product(U0, S0_sqrt, Phi);  % n1 x r x n3
B = transformed_t_product(V0, S0_sqrt, Phi);  % n2 x r x n3

%% -------------------- main loops --------------------
relChg = inf;

for iter = 1:maxIter
    At = A;  Bt = B;

    % current estimate
    Xt = transformed_t_product(At, transformed_tensor_ctranspose(Bt, Phi), Phi);

    % compute relative errors
    E = Xt - X;

    Error_fro(iter)  = norm(E(:)) / max(1, norm(X(:)));
    Error_spec(iter) = tensor_spec_norm_phi(E, Phi) / max(1, tensor_spec_norm_phi(X, Phi));

    % residual: res = M(Xt) - y
    res = op.forward(Xt) - y;

    % Rt = M^*(res)
    Rt = op.adjoint(res);

    Ah  = transformed_tensor_ctranspose(At, Phi);
    Bh  = transformed_tensor_ctranspose(Bt, Phi);

    Qt  = transformed_t_product(Ah, At, Phi) - transformed_t_product(Bh, Bt, Phi);  % r x r x n3

    GradA = transformed_t_product(Rt, Bt, Phi) ...
          + 0.5 * transformed_t_product(At, Qt, Phi);

    GradB = transformed_t_product(transformed_tensor_ctranspose(Rt, Phi), At, Phi) ...
          - 0.5 * transformed_t_product(Bt, Qt, Phi);

    % update
    A = At - eta * GradA;
    B = Bt - eta * GradB;

    % monitor change every 100 iters (same style as your matrix test)
    if iter == 1 || mod(iter,100)==0
        relChgA = norm(A(:)-At(:)) / max(1, norm(At(:)));
        relChgB = norm(B(:)-Bt(:)) / max(1, norm(Bt(:)));
        relChg  = max(relChgA, relChgB);
        fprintf('iter %4d | relChg %.3e | relRes %.3e\n', ...
            iter, relChg, norm(res)/y_norm);
    end

    if relChg < tol
        break;
    end
end

% trim to actual iterations
T = iter;
Error_fro  = Error_fro(1:T);
Error_spec = Error_spec(1:T);

%% -------------------- plot the results --------------------
figure(1); clf;
set(gcf, 'Color', 'w', 'Position', [100, 100, 1200, 380]);

% ----- data preparation -----
maxIter = min(maxIter, numel(Error_fro));  % safety
it   = 1:maxIter;

eps0 = 1e-16;   % avoid semilogy issues
eF   = max(Error_fro(1:maxIter),  eps0);
eS   = max(Error_spec(1:maxIter), eps0);

% ----- compact manual layout (1x2) -----
left = 0.07; right = 0.02; bottom = 0.18; top = 0.12; gap = 0.06;
W = (1 - left - right - gap)/2;
H = 1 - bottom - top;

axL = axes('Position', [left,           bottom, W, H]);   % linear
axR = axes('Position', [left+W+gap,     bottom, W, H]);   % log

% ================= Left panel: linear y-scale =================
hold(axL, 'on');

h1 = plot(axL, it, eF, '-o', ...
    'Color', [0.5, 0, 0.5], ...
    'LineWidth', 1.8, ...
    'MarkerSize', 7, ...
    'MarkerFaceColor', [0.5, 0, 0.5]);

h2 = plot(axL, it, eS, '-x', ...
    'Color', [1, 0.5, 0], ...
    'LineWidth', 1.8, ...
    'MarkerSize', 7);

xlim(axL, [1, it(end)]);
grid(axL, 'on'); grid(axL, 'minor');

xlabel(axL, 'Iteration', 'Interpreter', 'latex', 'FontSize', 14);
ylabel(axL, 'Relative error', 'Interpreter', 'latex', 'FontSize', 14);
title(axL, '(\uppercase\expandafter{\romannumeral 1}) Relative error vs. iteration (linear y)', 'Interpreter','latex', 'FontSize', 14);


set(axL, 'FontSize', 12, 'LineWidth', 1.1, ...
    'Box', 'on', 'TickLabelInterpreter', 'latex');

hold(axL, 'off');

% ================= Right panel: log y-scale =================
hold(axR, 'on');

h1r = semilogy(axR, it, eF, '-o', ...
    'Color', [0.5, 0, 0.5], ...
    'LineWidth', 1.8, ...
    'MarkerSize', 7, ...
    'MarkerFaceColor', [0.5, 0, 0.5]);

h2r = semilogy(axR, it, eS, '-x', ...
    'Color', [1, 0.5, 0], ...
    'LineWidth', 1.8, ...
    'MarkerSize', 7);

set(axR, 'YScale', 'log');
xlim(axR, [1, it(end)]);
grid(axR, 'on'); grid(axR, 'minor');

xlabel(axR, 'Iteration', 'Interpreter', 'latex', 'FontSize', 14);
ylabel(axR, '');   % <<< intentionally removed
title(axR, '(\uppercase\expandafter{\romannumeral 2}) Relative error vs. iteration (log y)', 'Interpreter','latex', 'FontSize', 14);

set(axR, 'FontSize', 12, 'LineWidth', 1.1, ...
    'Box', 'on', 'TickLabelInterpreter', 'latex');

hold(axR, 'off');

% ================= One shared legend =================
lgd = { ...
    '$\|\mathcal{A}_t *_\Phi \mathcal{B}_t^H-\mathcal{X}_\star\|_\mathrm{F}/\|\mathcal{X}_\star\|_\mathrm{F}$', ...
    '$\|\mathcal{A}_t *_\Phi \mathcal{B}_t^H-\mathcal{X}_\star\|_{2,\Phi}/\|\mathcal{X}_\star\|_{2,\Phi}$' ...
};

leg = legend(axR, lgd, ...
    'Location', 'northeast', ...
    'Interpreter', 'latex', ...
    'FontSize', 16, ...
    'Box', 'off');

% optional fine-tuning of legend position
p = get(leg, 'Position'); 
p(1) = p(1) + 0.01;
set(leg, 'Position', p);


