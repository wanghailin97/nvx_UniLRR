%% test the linear convergence of nvx_LRMR_GD
% Fro-norm and spectral-norm based relative error v.s. iteration

clear; close all;
addpath(genpath(pwd));
rng(0);

%% -------------------- simulated random low-rank matrix --------------------
n = 100;
n1 = n; n2 = n;
dim = [n1,n2];
r = 5;

X = randn(n,r) * randn(r,n);

%% -------------------- settings --------------------

c_m = 5;
m = ceil(c_m * n * r );
op = make_Gaussian_operator(dim, m);

y = op.forward(X);
y_norm = max(1, norm(y));

eta = 0.001;
maxIter = 1000;
tol = 1e-12;

Error_fro = zeros(maxIter, 1);
Error_spec = zeros(maxIter, 1);

%% -------------------- spectral init --------------------
X0 = op.adjoint(y);   % adjoint backprojection
[U,S,V] = svd(X0, 'econ');
U0 = U(:,1:r); S0 = S(1:r,1:r); V0 = V(:,1:r);
S0 = sqrt(max(S0,0));  
A  = U0*S0;
B  = V0*S0;

%% -------------------- main loops --------------------
for iter = 1:maxIter
    At = A;  Bt = B;
    Xt = At*Bt';
    
    % compute relative error
    Error_fro(iter) = norm(Xt- X, 'fro') / norm(X, 'fro');
    Error_spec(iter) = norm(Xt - X) / norm(X);
    
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
    
    if iter == 1 || mod(iter,100)==0
        relChgA = norm(A-At,'fro') / max(1, norm(At,'fro'));
        relChgB = norm(B-Bt,'fro') / max(1, norm(Bt,'fro'));
        relChg  = max(relChgA, relChgB);
        fprintf('iter %4d | relChg %.3e | relRes %.3e\n', ...
            iter, relChg, norm(res)/y_norm);
    end
    
    if relChg < tol
        break;
    end
end

%% plot the results
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
    '$\|\mathbf{A}_t\mathbf{B}_t^\top-\mathbf{X}_\star\|_\mathrm{F} / \|\mathbf{X}_\star\|_\mathrm{F}$', ...
    '$\|\mathbf{A}_t\mathbf{B}_t^\top-\mathbf{X}_\star\|_2 / \|\mathbf{X}_\star\|_2$' ...
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

%% optional export
% print(gcf, 'error_linear_vs_log.png', '-dpng', '-r300');
% print(gcf, 'error_linear_vs_log.pdf', '-dpdf');
