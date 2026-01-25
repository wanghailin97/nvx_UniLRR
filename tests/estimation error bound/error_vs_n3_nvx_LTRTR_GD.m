%% Test: final tensor error ||Xhat - Xstar||_F^2 versus n3
% Tensor size: n x n x n3, vary n3
% Fix n, tubal rank r, noise sigma
% Measurements: m = ceil(c * n * r * n3 * log(n))  (fixed rule across n3)

clear; close all;
addpath(genpath(pwd));
rng(0);

%% -------------------- fixed parameters --------------------
n = 60;                  % fixed spatial size
r = 3;                   % fixed tubal rank
sigma = 1e-2;            % fixed noise std
num_trials = 10;          % average across trials

% vary n3
n3_list = [3 4 5 6 8 10 12];   % modify as needed

% measurement rule (depends on n3)
c = 5;                   % oversampling constant

% GD params (fixed)
eta     = 1e-3;
maxIter = 2500;
tol     = 1e-10;

% storage
err2_mean = zeros(numel(n3_list),1);
err2_std  = zeros(numel(n3_list),1);
m_list    = zeros(numel(n3_list),1);

%% -------------------- main loop over n3 --------------------
for idx = 1:numel(n3_list)
    n3 = n3_list(idx);
    dim = [n, n, n3];

    % unitary Phi (depends on n3)
    Phi = dftmtx(n3)/sqrt(n3);

    % measurement number for this n3 (rule across n3)
    m = ceil(c * n * r * n3);
    m_list(idx) = m;

    errs = zeros(num_trials,1);

    for t = 1:num_trials
        rng(1000*idx + t);

        % ---- generate low-tubal-rank ground truth Xstar ----
        % optional normalization to keep ||Xstar||_F scale comparable across n3
        A0 = randn(n, r, n3);
        B0 = randn(n, r, n3);

        Xstar = transformed_t_product( ...
            A0, transformed_tensor_ctranspose(B0, Phi), Phi);

        % ---- Gaussian measurement operator ----
        op = make_Gaussian_operator(dim, m, 10*idx + t);

        % ---- noisy measurements ----
        y_clean = op.forward(Xstar);
        e = sigma * randn(m,1);
        y = y_clean + e;

        % ---- run solver ----
        [Xhat, ~, ~, info] = nvx_LTRTR_GD( ...
            y, op, dim, r, Phi, eta, maxIter, tol); 

        % ---- Frobenius error ----
        errs(t) = norm(Xhat(:) - Xstar(:));
        % normalized alternative:
        % errs(t) = norm(Xhat(:) - Xstar(:))^2 / max(1, norm(Xstar(:))^2);
    end

    err2_mean(idx) = mean(errs);
    err2_std(idx)  = std(errs);

    fprintf('n3=%3d, m=%d | mean err^2=%.3e (std=%.3e)\n', ...
        n3, m, err2_mean(idx), err2_std(idx));
end

%% -------------------- plot: error vs sqrt{n3} --------------------
figure(1); clf;
x = sqrt(n3_list(:));
errorbar(x, err2_mean, err2_std, '-o', ...
    'LineWidth', 1.6, 'MarkerSize', 7);
set(gca, 'YScale', 'log');   % log-scale often clearer
grid on; grid minor;

xlabel('$n_3$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\|\hat{\mathcal X}-\mathcal X_\star\|_F$', ...
       'Interpreter', 'latex', 'FontSize', 16);
title(sprintf(['nvx\\_LTRTR\\_GD: error vs $n_3$ ' ...
    '(n=%d, r=%d, \\sigma=%.1e, trials=%d)'], ...
    n, r, sigma, num_trials), ...
    'Interpreter', 'latex', 'FontSize', 12);

%% -------------------- plot: m vs n3 (sanity check) --------------------
figure(2); clf;
plot(n3_list(:), m_list, '-s', 'LineWidth', 1.6, 'MarkerSize', 7);
grid on; grid minor;
xlabel('$n_3$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$m=\lceil c n r n_3 \log n \rceil$', ...
       'Interpreter', 'latex', 'FontSize', 16);
title(sprintf('Measurements used (c=%.1f, n=%d, r=%d)', c, n, r), ...
    'Interpreter', 'latex', 'FontSize', 12);
