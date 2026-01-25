%% Test: final error ||Xhat - Xstar||_F versus sqrt(r)
% Fixed n, sigma, m = ceil(c * n * r * log(n))
% Using nvx_LRMR_GD
% Model: y = M(Xstar) + e, with Gaussian operator M

clear; close all;
addpath(genpath(pwd));
rng(0);

%% -------------------- fixed settings --------------------
n = 100;                 % fixed matrix dimension (n x n)
dim = [n, n];

sigma = 0.1;            % fixed noise std in measurement domain
c = 5;                   % oversampling constant in m = c n r log n
num_trials = 10;          % average across trials

% rank sweep (must satisfy r <= n)
r_list = 2:2:20;   % modify as you like (<= n)
r_list = r_list(r_list <= n);

eta_base = 1e-3;         % base stepsize; you can keep fixed
maxIter  = 3000;
tol      = 1e-10;

% storage
err2_mean = zeros(numel(r_list),1);
err2_std  = zeros(numel(r_list),1);
m_list    = zeros(numel(r_list),1);

%% -------------------- sweep over ranks --------------------
for idx = 1:numel(r_list)
    r = r_list(idx);

    % measurements count
    m = ceil(c * n * r);
    m_list(idx) = m;

    err2_trials = zeros(num_trials,1);

    for t = 1:num_trials
        rng(1000*idx + t);

        % ---- generate rank-r Xstar  ----
        Xstar = randn(n,r) * randn(r,n);

        % ---- Gaussian operator ----
        op = make_Gaussian_operator(dim, m, 10*idx + t);

        % ---- measurements with noise ----
        y_clean = op.forward(Xstar);
        e = sigma * randn(m,1);
        y = y_clean + e;

        % ---- run solver ----
        % (option) keep eta fixed or scale a bit with r
        eta = eta_base;  % simplest: fixed
        [Xhat, ~, ~, info] = nvx_LRMR_GD(y, op, dim, r, eta, maxIter, tol); 

        % ---- final squared Fro error ----
        err2_trials(t) = norm(Xhat - Xstar, 'fro');
    end

    err2_mean(idx) = mean(err2_trials);
    err2_std(idx)  = std(err2_trials);

    fprintf('r=%3d (sqrt=%.2f), m=%d | mean err^2=%.3e, std=%.3e\n', ...
        r, sqrt(r), m, err2_mean(idx), err2_std(idx));
end

%% -------------------- plot: error^2 vs sqrt(r) --------------------
figure(1); clf;
x = sqrt(r_list(:));

errorbar(x, err2_mean, err2_std, '-o', 'LineWidth', 1.6, 'MarkerSize', 7);
grid on; grid minor;
xlabel('$\sqrt{r}$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\left\Vert\hat{\mathbf X}-\mathbf X_\star\right\Vert_\mathrm{F}$', 'Interpreter', 'latex', 'FontSize', 16);
%title(sprintf('nvx\\_LRMR\\_GD: error vs $\\sqrt{r}$ (n=%d, \\sigma=%.1e, trials=%d, m=ceil(c n r log n))', ...
%    n, sigma, num_trials), 'Interpreter', 'latex', 'FontSize', 13);

