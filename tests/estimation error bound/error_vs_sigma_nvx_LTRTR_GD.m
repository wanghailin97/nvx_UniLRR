%% Test: final tensor error ||Xhat - Xstar||_F versus noise level sigma
% Fix tensor dimensions, tubal rank r, measurements m; vary sigma
% Using nvx_LTRTR_GD

clear; close all;
addpath(genpath(pwd));
rng(0);

%% -------------------- fixed parameters --------------------
n = 20;
n1 = n; n2 = n; 
n3 = 3;      % fixed tensor size
dim = [n1, n2, n3];

r = 2;                         % fixed tubal rank

% transform (unitary)
Phi = dftmtx(n3)/sqrt(n3);

% fixed measurement number (choose either fixed constant or a rule)
c = 5;
m = ceil(c * n1 * r * n3);   % fixed once, do NOT change with sigma

eta     = 1e-3;
maxIter = 3000;
tol     = 1e-10;

num_trials = 10;

% sigma sweep (log-spaced)
sigma_list = logspace(-4, -1, 8);   % e.g. 1e-4 ... 1e-1

% storage
err2_mean = zeros(numel(sigma_list),1);
err2_std  = zeros(numel(sigma_list),1);

%% -------------------- generate a fixed low-tubal-rank ground truth X_star --------------------
% Construct Xstar = A0 *_Phi B0^H
A0 = randn(n1, r, n3);      % real case; if you want complex, add +1i*randn(...)
B0 = randn(n2, r, n3);

Xstar = transformed_t_product(A0, transformed_tensor_ctranspose(B0, Phi), Phi);

%% -------------------- main loop over sigma --------------------
for sidx = 1:numel(sigma_list)
    sigma = sigma_list(sidx);

    errs = zeros(num_trials,1);

    for t = 1:num_trials
        rng(1000*sidx + t);

        % ---- measurement operator (Gaussian) ----
        op = make_Gaussian_operator(dim, m, 10*sidx + t);

        % ---- noisy measurements ----
        y_clean = op.forward(Xstar);
        e = sigma * randn(m,1);
        y = y_clean + e;

        % ---- run solver ----
        [Xhat, ~, ~, info] = nvx_LTRTR_GD(y, op, dim, r, Phi, eta, maxIter, tol); 

        % ---- Frobenius error ----
        errs(t) = norm(Xhat(:) - Xstar(:));

        % If you prefer normalized:
        % errs(t) = norm(Xhat(:) - Xstar(:))^2 / Xstar_fro2;
    end

    err2_mean(sidx) = mean(errs);
    err2_std(sidx)  = std(errs);

    fprintf('sigma=%.1e | mean err^2=%.3e (std=%.3e)\n', ...
        sigma, err2_mean(sidx), err2_std(sidx));
end

%% -------------------- plot: error vs sigma --------------------
figure(1); clf;

errorbar(sigma_list(:), err2_mean, err2_std, '-o', ...
    'LineWidth', 1.6, 'MarkerSize', 7);
set(gca, 'XScale', 'log', 'YScale', 'log');
grid on; grid minor;

xlabel('$\sigma$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\|\hat{\mathcal X}-\mathcal X_\star\|_F^2$', 'Interpreter', 'latex', 'FontSize', 16);
title(sprintf('nvx\\_LTRTR\\_GD: error vs noise (n1=%d,n2=%d,n3=%d, r=%d, m=%d, trials=%d)', ...
    n1, n2, n3, r, m, num_trials), 'Interpreter', 'latex', 'FontSize', 12);
