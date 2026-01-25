function test_phi_tensor_ops()
%TEST_PHI_TENSOR_OPS  Sanity checks for Phi-transform tensor ops

clc; clear; close all;
rng(0);

fprintf('==== Test for Phi-transform tensor ops ====\n');

% -----------------------------
% Settings
% -----------------------------
n1 = 7; m = 5; n2 = 6; n3 = 8;

% random complex tensors
A = randn(n1,m,n3) + 1i*randn(n1,m,n3);
B = randn(m,n2,n3) + 1i*randn(m,n2,n3);
X = randn(n1,n2,n3) + 1i*randn(n1,n2,n3);

% Unitary Phi (DFT)
Phi = dftmtx(n3) / sqrt(n3);

tol = 1e-10;

% -----------------------------
% Test 1: mode3_product invertibility
% -----------------------------
X_phi = mode3_product(X, Phi);
X_rec = mode3_product(X_phi, Phi');  % Phi^H

relerr1 = norm(X_rec(:) - X(:)) / max(1, norm(X(:)));
fprintf('[Test 1] Invertibility of mode3_product: relerr = %.3e\n', relerr1);
assert(relerr1 < 1e-9, 'Test 1 failed: mode3_product invertibility.');

% -----------------------------
% Test 2: (A *_Phi B)^H = B^H *_Phi A^H
% -----------------------------
C   = transformed_t_product(A, B, Phi);
Ch  = transformed_tensor_ctranspose(C, Phi);

Ah  = transformed_tensor_ctranspose(A, Phi);
Bh  = transformed_tensor_ctranspose(B, Phi);
rhs = transformed_t_product(Bh, Ah, Phi);

relerr2 = norm(Ch(:) - rhs(:)) / max(1, norm(Ch(:)));
fprintf('[Test 2] (A*_Phi B)^H = B^H *_Phi A^H: relerr = %.3e\n', relerr2);
assert(relerr2 < 1e-9, 'Test 2 failed: t-product conjugate transpose rule.');

% -----------------------------
% Helpers for transform-domain reconstruction check
% -----------------------------
% Given V (n2 x n2 x n3) or (n2 x r x n3), build V^H page-wise in transform domain
build_Vh_phi = @(Vphi) permute(conj(Vphi), [2 1 3]);

% -----------------------------
% Test 3: Full transformed_tsvd reconstruction (transform domain)
% -----------------------------
[Ufull, Sfull, Vfull] = transformed_tsvd(X, Phi, true);

X_Phi  = mode3_product(X, Phi);
U_Phi  = mode3_product(Ufull, Phi);
S_Phi  = mode3_product(Sfull, Phi);
V_Phi  = mode3_product(Vfull, Phi);

Vh_Phi = build_Vh_phi(V_Phi);

X_Phi_rec = page_multiply3(U_Phi, S_Phi, Vh_Phi);

relerr3 = norm(X_Phi_rec(:) - X_Phi(:)) / max(1, norm(X_Phi(:)));
fprintf('[Test 3] transformed_tsvd (full) recon in Phi-domain: relerr = %.3e\n', relerr3);
assert(relerr3 < 1e-8, 'Test 3 failed: transformed_tsvd reconstruction.');

% -----------------------------
% Test 4: Skinny transformed_tsvd_skinny reconstruction (transform domain)
% -----------------------------
r = min(n1,n2); % exact skinny rank, should reconstruct exactly in transform domain
[Usk, Ssk, Vsk] = transformed_tsvd_skinny(X, Phi, r);

Usk_Phi = mode3_product(Usk, Phi); % n1 x r x n3
Ssk_Phi = mode3_product(Ssk, Phi); % r  x r x n3
Vsk_Phi = mode3_product(Vsk, Phi); % n2 x r x n3

Vskh_Phi = build_Vh_phi(Vsk_Phi);  % r x n2 x n3

X_Phi_rec2 = page_multiply3(Usk_Phi, Ssk_Phi, Vskh_Phi);

relerr4 = norm(X_Phi_rec2(:) - X_Phi(:)) / max(1, norm(X_Phi(:)));
fprintf('[Test 4] transformed_tsvd_skinny (exact r) recon in Phi-domain: relerr = %.3e\n', relerr4);
assert(relerr4 < 1e-8, 'Test 4 failed: transformed_tsvd_skinny reconstruction.');

fprintf('\nAll tests PASSED ?\n');

end

% ============================================================
% Local helper: page-wise multiply of three tensors
% C(:,:,k) = A(:,:,k) * B(:,:,k) * D(:,:,k)
% Supports either pagemtimes (fast) or fallback loop.
% ============================================================
function C = page_multiply3(A, B, D)
% A: n1 x p x n3
% B: p  x q x n3
% D: q  x n2 x n3
[n1, ~, n3] = size(A);
[~, n2, ~]  = size(D);

if exist('pagemtimes','file') == 2
    C = pagemtimes(pagemtimes(A, B), D);
else
    C = zeros(n1, n2, n3, 'like', A);
    for k = 1:n3
        C(:,:,k) = A(:,:,k) * B(:,:,k) * D(:,:,k);
    end
end
end
