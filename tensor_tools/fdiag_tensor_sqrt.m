function S_sqrt = fdiag_tensor_sqrt(S, Phi)
[r1, r2, n3] = size(S);
assert(r1 == r2, 'S must be square in its first two modes.');
assert(isequal(size(Phi), [n3 n3]), 'Phi size mismatch in fdiag_tensor_sqrt.');

S_Phi = mode3_product(S, Phi);                       % r x r x n3
Ssq_Phi = zeros(r1, r2, n3, 'like', S_Phi);

for k = 1:n3
    Sk = S_Phi(:,:,k);
    d  = real(diag(Sk));                             % should be real nonnegative ideally
    d  = max(d, 0);                                  % numerical safety
    Ssq_Phi(:,:,k) = diag(sqrt(d));
end

S_sqrt = mode3_product(Ssq_Phi, Phi');               % inverse transform
end