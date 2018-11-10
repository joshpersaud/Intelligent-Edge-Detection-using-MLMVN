
%Weights adjustment for a hidden neuron

%function w_adj = HidNeuron_weightadj(X, w, delta, N)
function w_adj = HidNeuron_weightadj(X, X_pinv, w, delta, N)
%function w_adj = HidNeuron_weightadj(X_pinv, w, delta, N)


%X = matrix of hidden neuron inputs (N x n), where N=number of learning
%samples, n = number of input variables

%w = (n+1 x 1) vector of weights of the hidden neuron

%delta = (N x 1) vector of errors

%X_pinv = pre-computed pseudo-inverse of the matrix of network inputs


%N = size(X, 1);

%append a column of 1s to X from the left, making it a (N x n+1) matrix
col_app(1:N) = 1;
col_app = col_app.';
X = [col_app X];

%compute the weighted sums
z_c = X * w;

z_c_mag = abs(z_c);

%Use LLS to compute a weights adjustment vector adj_val
%adj_vec = X \ delta;

%Compute the full SVD of X
%[U,S,V] = svd(X);

%M = n+1
M = length(w);

%Retain only the first M columns of U, and first M rows of S
%U_hat = U(:, 1:M);
%S_hat = S(1:M, :); %S_hat becomes an M x M square matrix

%Construct the pseudo-inverse of S
%S_hpinv = diag(1 ./ diag(S_hat));

%Construct the pseudo-inverse of X
%X_pinv = V * S_hpinv * U_hat';

%LLS: apply X_pinv to delta
adj_vec = X_pinv * (1 .* delta); 

%the new weights are given by 
w_adj = w + adj_vec;




