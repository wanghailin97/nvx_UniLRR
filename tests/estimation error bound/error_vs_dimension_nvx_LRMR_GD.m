%% Test: final recovery error ||Xhat - Xstar||_F versus sqrt(n)
% Fixed r=1 for simplicity, sigma=0.1
% Using nvx_LRMR_GD 
% Model: y = M(Xstar) + e, with Gaussian operator M

clear; close all;
addpath(genpath(pwd));
rng(0);

%% -------------------- experiment settings --------------------
r = 1;                 % fixed rank
sigma = 1e-1;          % fixed noise std (per measurement)
num_trials = 10;        % average over trials for stability

% choose a range of n (matrix size n x n)
%n_list = [40 60 80 100 140 200];   % you can modify
n_list = (6:15).^2;

% measurement rule (keep "noise level fixed"):
% Here we scale m with degrees of freedom ~ n r log n (typical)
c_m = 5;  % oversampling constant
eta = 1e-3;
maxIter = 2000;
tol = 1e-10;

% storage
err2_mean = zeros(numel(n_list),1);
err2_std  = zeros(numel(n_list),1);
m_list    = zeros(numel(n_list),1);

%% -------------------- main loop --------------------
for idx = 1:numel(n_list)
    n = n_list(idx);
    dim = [n, n];

    % measurements count (fix rule across n)
    m = ceil(c_m * n * r);
    m_list(idx) = m;

    err2_trials = zeros(num_trials,1);

    for t = 1:num_trials
        rng(1000*idx + t);  % reproducibility per (n,trial)

        % ---- generate X_star ----
        Xstar = randn(n,r) * randn(r,n);
		
        % ---- Gaussian measurement operator ----
        opM = make_Gaussian_operator(dim, m, 10*idx + t);

        % ---- noisy measurements ----
        y_clean = opM.forward(Xstar);
        e = sigma * randn(m,1);
        y = y_clean + e;

        % ---- run solver ----
        [Xhat, ~, ~, info] = nvx_LRMR_GD(y, opM, dim, r, eta, maxIter, tol); 

        % ---- final error (Frobenius) ----
        err2_trials(t) = norm(Xhat - Xstar, 'fro');
    end

    err2_mean(idx) = mean(err2_trials);
    err2_std(idx)  = std(err2_trials);

    fprintf('n=%4d (sqrt=%.2f), m=%d | mean err^2=%.3e, std=%.3e\n', ...
        n, sqrt(n), m, err2_mean(idx), err2_std(idx));
end

%% -------------------- plot: error vs sqrt(n) --------------------
figure(1); clf;
x = sqrt(n_list(:));

errorbar(x, err2_mean, err2_std, '-o', 'LineWidth', 1.6, 'MarkerSize', 7);
grid on; grid minor;
xlabel('$\sqrt{n}$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\left\Vert\hat{\mathbf X}-\mathbf X_\star\right\Vert_\mathrm{F}$', 'Interpreter', 'latex', 'FontSize', 16);

