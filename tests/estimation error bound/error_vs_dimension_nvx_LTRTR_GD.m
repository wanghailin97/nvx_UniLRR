%% Test: final tensor error ||Xhat - Xstar||_F versus n
% Tensor size: n x n x n3, vary n
% Fix n3, tubal rank r, noise sigma% Measurements: m = ceil(c * n * r * n3 * log(n))   (fixed rule across n)

clear; close all;
addpath(genpath(pwd));
rng(0);

%% -------------------- fixed parameters --------------------
n3 = 3;                  % fixed 3rd-mode length
r  = 1;                  % fixed tubal rank
sigma = 0.1;            % fixed noise std
num_trials = 10;          % average across trials

% vary n
n_list = (6:15).^2;  % modify as you like

% measurement rule
c = 5;   % oversampling constant

% GD params (fixed)
eta     = 1e-3;
maxIter = 2500;
tol     = 1e-10;

% storage
err2_mean = zeros(numel(n_list),1);
err2_std  = zeros(numel(n_list),1);
m_list    = zeros(numel(n_list),1);

%% -------------------- main loop over n --------------------
for idx = 1:numel(n_list)
    n = n_list(idx);
    dim = [n, n, n3];

    % unitary Phi (depends only on n3)
    Phi = dftmtx(n3)/sqrt(n3);

    % measurement number for this n (rule across n)
    m = ceil(c * n * r * n3);
    m_list(idx) = m;

    errs = zeros(num_trials,1);

    for t = 1:num_trials
        rng(1000*idx + t);

        % ---- generate low-tubal-rank ground truth Xstar ----
        A0 = randn(n, r, n3);
        B0 = randn(n, r, n3);
        Xstar = transformed_t_product(A0, transformed_tensor_ctranspose(B0, Phi), Phi);

        % ---- Gaussian measurement operator ----
        opM = make_Gaussian_operator(dim, m, 10*idx + t);

        % ---- noisy measurements ----
        y_clean = opM.forward(Xstar);
        e = sigma * randn(m,1);
        y = y_clean + e;

        % ---- run solver ----
        [Xhat, ~, ~, info] = nvx_LTRTR_GD(y, opM, dim, r, Phi, eta, maxIter, tol);

        % ---- Frobenius error ----
        errs(t) = norm(Xhat(:) - Xstar(:));
        % If you prefer normalized:
        % errs(t) = norm(Xhat(:) - Xstar(:))^2 / max(1, norm(Xstar(:))^2);
    end

    err2_mean(idx) = mean(errs);
    err2_std(idx)  = std(errs);

    fprintf('n=%4d (sqrt=%.2f), m=%d | mean err^2=%.3e (std=%.3e)\n', ...
        n, sqrt(n), m, err2_mean(idx), err2_std(idx));
end

%% -------------------- plot: error vs sqrt(n) --------------------
figure(1); clf;
x = sqrt(n_list(:));
errorbar(x, err2_mean, err2_std, '-s', 'LineWidth', 1.6, 'MarkerSize', 7);
set(gca, 'YScale', 'log');
grid on; grid minor;
xlabel('$\sqrt{n}$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\|\hat{\mathcal X}-\mathcal X_\star\|_F^2$', 'Interpreter', 'latex', 'FontSize', 16);

