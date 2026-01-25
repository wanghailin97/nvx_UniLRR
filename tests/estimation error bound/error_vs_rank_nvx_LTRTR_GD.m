%% Test: final tensor error ||Xhat - Xstar||_F^2 versus sqrt(r)
% Tensor size: n x n x n3, vary tubal rank r
% Fix n, n3, noise sigma
% Measurements: m = ceil(c * n * r * n3 * log(n))

clear; close all;
addpath(genpath(pwd));
rng(0);

%% -------------------- fixed parameters --------------------
n  = 100;                 % fixed spatial size
n3 = 2;                  % fixed 3rd-mode length
dim = [n, n, n3];

sigma = 0.1;            % fixed noise std
num_trials = 5;          % average across trials

% rank sweep (must satisfy r <= n)
r_list = 2:2:20; 
r_list = r_list(r_list <= n);

% measurement rule (depends on r)
c = 5;

% transform (depends only on n3)
Phi = dftmtx(n3)/sqrt(n3);

% GD params (fixed)
eta     = 1e-2;
maxIter = 3000;
tol     = 1e-10;

% storage
err2_mean = zeros(numel(r_list),1);
err2_std  = zeros(numel(r_list),1);
m_list    = zeros(numel(r_list),1);

%% -------------------- main loop over r --------------------
for idx = 1:numel(r_list)
    r = r_list(idx);

    % measurement number for this r
    m = ceil(c * n * r * n3);
    m_list(idx) = m;

    errs = zeros(num_trials,1);

    for t = 1:num_trials
        rng(1000*idx + t);

        % ---- generate low-tubal-rank ground truth Xstar ----
        % Optional scaling to keep ||Xstar||_F comparable across r:
        % dividing by sqrt(r) prevents energy blow-up with rank.
        A0 = randn(n, r, n3) / sqrt(r);
        B0 = randn(n, r, n3) / sqrt(r);

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

        % ---- squared Frobenius error ----
        errs(t) = norm(Xhat(:) - Xstar(:));
        % normalized alternative:
        % errs(t) = norm(Xhat(:) - Xstar(:))^2 / max(1, norm(Xstar(:))^2);
    end

    err2_mean(idx) = mean(errs);
    err2_std(idx)  = std(errs);

    fprintf('r=%3d (sqrt=%.2f), m=%d | mean err^2=%.3e (std=%.3e)\n', ...
        r, sqrt(r), m, err2_mean(idx), err2_std(idx));
end

%% -------------------- plot: error^2 vs sqrt(r) --------------------
figure(1); clf;
x = sqrt(r_list(:));

errorbar(x, err2_mean, err2_std, '-o', ...
    'LineWidth', 1.6, 'MarkerSize', 7);
%set(gca, 'YScale', 'log');   % often informative
grid on; grid minor;

xlabel('$\sqrt{r}$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\|\hat{\mathcal X}-\mathcal X_\star\|_F$', ...
       'Interpreter', 'latex', 'FontSize', 16);
title(sprintf(['nvx\\_LTRTR\\_GD: error vs $\\sqrt{r}$ ' ...
    '(n=%d, n_3=%d, \\sigma=%.1e, trials=%d)'], ...
    n, n3, sigma, num_trials), ...
    'Interpreter', 'latex', 'FontSize', 12);

%% -------------------- plot: m vs r (sanity) --------------------
figure(2); clf;
plot(r_list(:), m_list, '-s', 'LineWidth', 1.6, 'MarkerSize', 7);
grid on; grid minor;

xlabel('$r$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$m=\lceil c n r n_3 \log n \rceil$', ...
       'Interpreter', 'latex', 'FontSize', 16);
title(sprintf('Measurements used (c=%.1f, n=%d, n3=%d)', c, n, n3), ...
    'Interpreter', 'latex', 'FontSize', 12);
