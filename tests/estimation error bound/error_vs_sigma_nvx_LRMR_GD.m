%% Test: final error ||Xhat - Xstar||_F^2 versus noise level sigma
% Fix n, r, m; vary sigma
% Using nvx_LRMR_GD

clear; close all;
addpath(genpath(pwd));
rng(0);

%% -------------------- fixed parameters --------------------
n = 20;                 % matrix size n x n
r = 2;                   % fixed rank
dim = [n, n];

% fix m (you can choose either a fixed number or a rule)
c = 5;
m = ceil(c * n * r);    % fixed once (do NOT change with sigma)

eta     = 1e-3;
maxIter = 3000;
tol     = 1e-10;

num_trials = 10;         % average over trials for stability

% sigma sweep (log-spaced is typical)
sigma_list = logspace(-4, -1, 8);   % e.g. 1e-4 ... 1e-1

% storage
err2_mean = zeros(numel(sigma_list),1);
err2_std  = zeros(numel(sigma_list),1);

%% -------------------- generate a fixed ground truth X_star (optional) --------------------
Xstar = randn(n,r) * randn(r,n);

%% -------------------- main loop over sigma --------------------
for sidx = 1:numel(sigma_list)
    sigma = sigma_list(sidx);

    errs = zeros(num_trials,1);

    for t = 1:num_trials
        rng(1000*sidx + t);

        % ---- measurement operator (fix distribution; resample per trial) ----
        op = make_Gaussian_operator(dim, m, 10*sidx + t);

        % ---- measurements with noise ----
        y_clean = op.forward(Xstar);
        e = sigma * randn(m,1);
        y = y_clean + e;

        % ---- run solver ----
        [Xhat, ~, ~, info] = nvx_LRMR_GD(y, op, dim, r, eta, maxIter, tol);

        % ---- squared Frobenius error ----
        errs(t) = norm(Xhat - Xstar, 'fro');
    end

    err2_mean(sidx) = mean(errs);
    err2_std(sidx)  = std(errs);

    fprintf('sigma=%.1e | mean err^2=%.3e (std=%.3e)\n', ...
        sigma, err2_mean(sidx), err2_std(sidx));
end

%% -------------------- plot: error^2 vs sigma --------------------
figure(1); clf;

ax = gca;
hold(ax, 'on');

errorbar(sigma_list(:), err2_mean, err2_std, '-o', ...
    'LineWidth', 1.8, 'MarkerSize', 7);
set(gca, 'XScale', 'log', 'YScale', 'log');   % log-log is usually most informative
grid(ax, 'on'); 
grid(ax, 'minor');
ax.XLim   = [1e-4, 1e-1];
% lgd = {
%     '$\|\mathbf{A}_t\mathbf{B}_t^\top-\mathbf{X}_\star\|_\mathrm{F} / \|\mathbf{X}_\star\|_\mathrm{F}$'
% };
% legend(ax, lgd, 'Location', 'northeast', ...
%     'Interpreter', 'latex', 'FontSize', 20, 'Box', 'off');

xlabel('$\sigma$', 'Interpreter', 'latex', 'FontSize', 18);
ylabel('$\|\hat{\mathbf A}\hat{\mathbf B}^\top-\mathbf X_\star\|_\mathrm{F}$', 'Interpreter', 'latex', 'FontSize', 18);
title(ax, 'Error v.s. $\sigma$', 'Interpreter', 'latex', 'FontSize', 18);
set(ax, 'FontSize', 14, 'LineWidth', 1.2);
hold(ax, 'off');