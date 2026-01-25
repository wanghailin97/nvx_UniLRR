function val = tensor_spec_norm_phi(X, Phi)
% Phi-spectral norm: ||X||_{2,Phi} := max_k ||X_Phi(:,:,k)||_2
X_Phi = mode3_product(X, Phi);
n3 = size(X_Phi,3);
val = 0;
for k = 1:n3
    val = max(val, norm(X_Phi(:,:,k), 2));
end
end